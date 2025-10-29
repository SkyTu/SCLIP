#pragma once

#include "mpc/mpc.h"
#include "mpc/fix_tensor.h"
#include "mpc/tensor_ops.h"
#include "mpc/truncate.h"
#include "mpc/secure_tensor_ops.h"
#include "mpc/elementwise_mul.h"
#include "mpc/matmul.h"
#include "utils/random.h"

// A struct to hold all the dimension and configuration parameters for the L2Norm layer.
struct L2NormLayerParams {
    int B;       // Batch size
    int in_dim;  // Input dimension
};

template <typename T, int BW, int smallBW, int F, int K_INT>
struct L2NormRandomness {
    SquareRandomness<T, BW, smallBW, F, K_INT, 3> square_randomness;
    ZeroExtendRandomness<T, BW, smallBW, F, K_INT, 2> zero_extend_randomness_square;
    InvSqrtRandomness<T, BW, smallBW, F, K_INT, 2> inv_sqrt_randomness;
    ElementwiseMulRandomness<T, BW, smallBW, F, K_INT, 3> elementwise_mul_randomness;
    ZeroExtendRandomness<T, BW, smallBW, F, K_INT, 3> zero_extend_randomness_elemul;
};

// IN_BW: n-f
// OUT_BW: n
template <typename T, int IN_BW, int OUT_BW, int F, int K_INT>
class L2NormLayer {
public:
    using InputTensor = FixTensor<T, IN_BW, F, K_INT, 2>;
    using OutputTensor = FixTensor<T, OUT_BW, F, K_INT, 3>;
    using IncomingGradTensor = FixTensor<T, IN_BW, F, K_INT, 2>;
    using OutgoingGradTensor = FixTensor<T, OUT_BW, F, K_INT, 2>;
    // For the norm, which is a vector of scalars (one for each item in batch)
    using NormTensor = FixTensor<T, IN_BW, F, K_INT, 2>;

    L2NormLayerParams p;
    L2NormRandomness<T, OUT_BW, IN_BW, F, K_INT> randomness;
    // To store values for the backward pass
    InputTensor image_share;
    InputTensor text_share;
    InputTensor image_rec;
    InputTensor text_rec;
    NormTensor norm_rec; // Store the reconstructed norm of each vector in the batch

public:
    L2NormLayer(const L2NormLayerParams& params) : p(params) {}

    size_t get_randomness_size() {
        size_t total_size = 0;
        total_size += get_square_random_size<T, 3>(2, p.B, p.in_dim);
        total_size += get_zero_extend_random_size<T, 2>(-1, 2, p.B);
        total_size += get_inv_sqrt_random_size<T, 2>(-1, 2, p.B);
        total_size += get_elementwise_mul_random_size<T, 3>(2, p.B, p.in_dim);
        total_size += get_zero_extend_random_size<T, 3>(2, p.B, p.in_dim);
        return total_size;
    }

    void read_randomness(MPC& mpc) {
        randomness.square_randomness = read_square_randomness<T, OUT_BW, IN_BW, F, K_INT, 3>(mpc, 2, p.B, p.in_dim);
        randomness.zero_extend_randomness_square = read_zero_extend_randomness<T, OUT_BW, IN_BW, F, K_INT, 2>(mpc, -1, 2, p.B);
        randomness.inv_sqrt_randomness = read_inv_sqrt_randomness<T, OUT_BW, IN_BW, F, K_INT, 2>(mpc, -1, 2, p.B);
        randomness.elementwise_mul_randomness = read_elementwise_mul_randomness<T, OUT_BW, IN_BW, F, K_INT, 3>(mpc, 2, p.B, p.in_dim);
        randomness.zero_extend_randomness_elemul = read_zero_extend_randomness<T, OUT_BW, IN_BW, F, K_INT, 3>(mpc, 2, p.B, p.in_dim);
    }
    
    
    void generate_randomness(Buffer& p0_buf, Buffer& p1_buf) {
        // generate_square_randomness<T, IN_BW, F, K_INT, 2>(p0_buf, p1_buf);
        FixTensor<T, IN_BW, F, K_INT, 3> R(2, p.B, p.in_dim);
        FixTensor<T, OUT_BW, F, K_INT, 3> R_N(2, p.B, p.in_dim);
        FixTensor<T, OUT_BW, F, K_INT, 3> R_SQUARE(2, p.B, p.in_dim);
        FixTensor<T, OUT_BW, F, K_INT, 3> R_MSB(2, p.B, p.in_dim);
        FixTensor<T, OUT_BW, F, K_INT, 3> R_R_MSB(2, p.B, p.in_dim);
        R.setRandom();
        // R.setConstant(Fix<T,m,f,k>(0));
        R_N = extend_locally<OUT_BW, F, K_INT>(R);
        R_SQUARE = R_N * R_N;
        R_MSB = get_msb<OUT_BW, F, K_INT>(R_N);
        R_R_MSB = R_N * R_MSB;
        secret_share_and_write_tensor(R, p0_buf, p1_buf);
        secret_share_and_write_tensor(R_N, p0_buf, p1_buf);
        secret_share_and_write_tensor(R_SQUARE, p0_buf, p1_buf);
        secret_share_and_write_tensor(R_MSB, p0_buf, p1_buf);
        secret_share_and_write_tensor(R_R_MSB, p0_buf, p1_buf);
        
        generate_zero_extend_randomness<T, OUT_BW, IN_BW, F, K_INT, 2>(p0_buf, p1_buf, -1, 2, p.B);
        generate_inv_sqrt_randomness<T, OUT_BW, IN_BW, F, K_INT, 2>(p0_buf, p1_buf, -1, 2, p.B);
        FixTensor<T, IN_BW, F, K_INT, 3> r_x_m(2, p.B, p.in_dim), r_y_m(2, p.B, p.in_dim);
        FixTensor<T, OUT_BW, F, K_INT, 3> r_x_n(2, p.B, p.in_dim), r_y_n(2, p.B, p.in_dim), r_x_msb(2, p.B, p.in_dim), r_y_msb(2, p.B, p.in_dim);
        FixTensor<T, OUT_BW, F, K_INT, 3> r_xy(2, p.B, p.in_dim), r_x_rymsb(2, p.B, p.in_dim), r_xmsb_y(2, p.B, p.in_dim);
        r_x_m = R;
        Random rg;
        for (int i = 0; i < 2; i++) {
            for (int j = 0; j < p.B; j++) {
                T val = rg.template randomGE<T>(1, IN_BW)[0];
                for (int k = 0; k < p.in_dim; k++) {
                    r_y_m(i, j, k) = Fix<T, IN_BW, F, K_INT>(0);
                }
            }
        }
        // r_y_m.setRandom();
        r_x_n = extend_locally<OUT_BW, F, K_INT>(r_x_m);
        r_y_n = extend_locally<OUT_BW, F, K_INT>(r_y_m);
        r_x_msb = get_msb<OUT_BW, F, K_INT>(r_x_n);
        r_y_msb = get_msb<OUT_BW, F, K_INT>(r_y_n);
        r_xy = r_x_n * r_y_n;
        r_x_rymsb = r_x_n * r_y_msb;
        r_xmsb_y = r_x_msb * r_y_n;
        secret_share_and_write_tensor(r_x_m, p0_buf, p1_buf);
        secret_share_and_write_tensor(r_y_m, p0_buf, p1_buf);
        secret_share_and_write_tensor(r_x_n, p0_buf, p1_buf);
        secret_share_and_write_tensor(r_y_n, p0_buf, p1_buf);
        secret_share_and_write_tensor(r_x_msb, p0_buf, p1_buf);
        secret_share_and_write_tensor(r_y_msb, p0_buf, p1_buf);
        secret_share_and_write_tensor(r_xy, p0_buf, p1_buf);
        secret_share_and_write_tensor(r_x_rymsb, p0_buf, p1_buf);
        secret_share_and_write_tensor(r_xmsb_y, p0_buf, p1_buf);

        generate_zero_extend_randomness<T, OUT_BW, IN_BW, F, K_INT, 3>(p0_buf, p1_buf, 2, p.B, p.in_dim);
    }

    OutputTensor forward(const InputTensor& image_share, const InputTensor& text_share) {
        if (mpc_instance == nullptr) throw std::runtime_error("MPC instance must be initialized before using forward.");
        
        this->image_share = image_share;
        this->text_share = text_share;
        FixTensor<T, IN_BW, F, K_INT, 3> all_share(2, p.B, p.in_dim);
        for(int i = 0; i < p.B; i++) {
            for(int j = 0; j < p.in_dim; j++) {
                all_share(0, i, j) = image_share(i, j);
                all_share(1, i, j) = text_share(i, j);
            }
        }
        auto tmpall = reconstruct_tensor(all_share);
        std::cout << "all_share reconstructed is " << tmpall << std::endl;
        FixTensor<T, IN_BW, F, K_INT, 3> all_rec = all_share + randomness.square_randomness.R;
        all_rec = reconstruct_tensor(all_rec);
        // --- L2 Normalization Forward---
        // y = x / ||x||_2
        // Operations are performed row-wise for the batch.

        // 1. x_squared = x * x (element-wise)
        // TODO: Implement secure_elemwise_mul if not available
        // x_share bw: n - f, x_square bw: 
        FixTensor<T, OUT_BW, F, K_INT, 3> all_square = square_tensor_opt<T, OUT_BW, IN_BW, F, K_INT, 3, Eigen::RowMajor>(all_rec, randomness.square_randomness, true);
        FixTensor<T, IN_BW, F, K_INT, 3> all_square_m = truncate_reduce_tensor(all_square);
        auto tmp1 = reconstruct_tensor(all_square_m);
        std::cout << "all_square reconstructed is " << tmp1 << std::endl;
        // 2. sum_sq = sum(x_squared) row-wise
        // TODO: Implement sum_rows for FixTensor
        // This will result in a tensor of shape (2, B)
        FixTensor<T, IN_BW, F, K_INT, 2> all_sum_sq_m = sum_reduce_tensor<2, T, IN_BW, F, K_INT, Eigen::RowMajor>(all_square_m);
        auto tmpsum = reconstruct_tensor(all_sum_sq_m);
        std::cout << "all_sum_sq_m reconstructed is " << tmpsum << std::endl;
        FixTensor<T, OUT_BW, F, K_INT, 2> all_sum_sq = zero_extend_tensor<T, OUT_BW, IN_BW, F, K_INT, 2>(all_sum_sq_m, randomness.zero_extend_randomness_square);
        auto tmpext = reconstruct_tensor(all_sum_sq);
        std::cout << "all_sum_sq reconstructed is " << tmpext << std::endl;
        // 3. norm = 1/sqrt(sum_sq)
        // TODO: Implement secure_sqrt for MPC
        // This is a complex protocol. For now, we can use a placeholder.
        // shape: (2, B)
        FixTensor<T, IN_BW, F, K_INT, 2> norm = inv_sqrt_tensor<T, OUT_BW, IN_BW, F, K_INT, 2>(all_sum_sq, randomness.inv_sqrt_randomness);
        auto tmpnorm = reconstruct_tensor(norm);
        std::cout << "norm reconstructed is " << tmpnorm << std::endl;
        std::cout << "finished inv sqrt tensor" << std::endl;
        FixTensor<T, IN_BW, F, K_INT, 2> r_y_m_2d(2, p.B);
        for (int i = 0; i < 2; i++) {
            for (int j = 0; j < p.B; j++) {
                r_y_m_2d(i, j) = randomness.elementwise_mul_randomness.r_y_m(i, j, 0);
            }
        }
        norm = norm + r_y_m_2d;
        this->norm_rec = reconstruct_tensor(norm);
        std::cout << "norm reconstructed is " << this->norm_rec << std::endl;
        // 4. y = x * 1 / norm (element-wise)
        // Need to broadcast norm_rec to match shape of x_share
        FixTensor<T, IN_BW, F, K_INT, 3> norm_broadcasted(2, p.B, p.in_dim);
        for (int i = 0; i < 2; i++) {
            for (int j = 0; j < p.B; j++) {
                for (int k = 0; k < p.in_dim; k++) {
                    norm_broadcasted(i, j, k) = this->norm_rec(i, j);
                }
            }
        }
        std::cout << "norm_broadcasted is " << norm_broadcasted << std::endl;
        std::cout << "all_rec size: " << all_rec.size() << std::endl;
        FixTensor<T, OUT_BW, F, K_INT, 3> y = elementwise_mul_opt<T, OUT_BW, IN_BW, F, K_INT, 3>(all_rec, norm_broadcasted, randomness.elementwise_mul_randomness, true, true);
        std::cout << "finished elementwise mul tensor" << std::endl;   
        auto tmptmp = reconstruct_tensor(y);
        std::cout << "y reconstructed is " << tmptmp << std::endl;
        FixTensor<T, IN_BW, F, K_INT, 3> y_m = truncate_reduce_tensor(y);
        auto tmpy = reconstruct_tensor(y_m);
        std::cout << "y_m reconstructed is " << tmpy << std::endl;
        y = zero_extend_tensor<T, OUT_BW, IN_BW, F, K_INT, 3>(y_m, randomness.zero_extend_randomness_elemul);
        auto tmpzero = reconstruct_tensor(y);
        std::cout << "y reconstructed is " << tmpzero << std::endl;
        std::cout << "finished zero extend tensor" << std::endl;
        return y;
    }

    // auto backward(const IncomingGradTensor& incoming_grad_share) {
    //     if (mpc_instance == nullptr) throw std::runtime_error("MPC instance must be initialized before using backward.");
        
    //     // --- L2 Normalization Backward ---
    //     // dL/dx = (1/||x||) * ( dL/dy - y * ((dL/dy) . y) )
    //     // where y = x / ||x|| is the output from forward pass.
        
    //     // Broadcast the saved norm for calculations.
    //     auto norm_broadcasted = this->norm_rec.broadcast(Eigen::array<int, 2>{1, p.in_dim});

    //     // Recompute y_share = x / ||x||
    //     // TODO: Replace with secure division by public value
    //     auto y_share = input_share.array() / norm_broadcasted.array();

    //     // (dL/dy) . y (dot product per row)
    //     // TODO: Replace with secure element-wise multiplication
    //     auto dot_prod_elements_share = incoming_grad_share.array() * y_share.array();
        
    //     // TODO: Implement sum_rows for FixTensor
    //     auto dot_prod_share = static_cast<InputTensor>(dot_prod_elements_share).sum(Eigen::array<int, 1>{1});
        
    //     // Broadcast the dot product result for element-wise multiplication with y.
    //     auto dot_prod_broadcasted = dot_prod_share.broadcast(Eigen::array<int, 2>{1, p.in_dim});

    //     // y * ((dL/dy) . y)
    //     // TODO: Replace with secure element-wise multiplication
    //     auto term2_share = y_share.array() * dot_prod_broadcasted.array();
        
    //     // dL/dy - y * ((dL/dy) . y)
    //     auto parenthesis_term_share = incoming_grad_share.array() - term2_share;

    //     // (1/||x||) * (...)
    //     // TODO: Replace with secure division by public value
    //     auto outgoing_grad_share = parenthesis_term_share / norm_broadcasted.array();

    //     return static_cast<OutgoingGradTensor>(outgoing_grad_share);
    // }
};
