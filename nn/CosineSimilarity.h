#pragma once

#include "mpc/mpc.h"
#include "mpc/fix_tensor.h"
#include "mpc/tensor_ops.h"
#include "mpc/truncate.h"
#include "mpc/secure_tensor_ops.h"
#include "mpc/elementwise_mul.h"
#include "mpc/matmul.h"
#include "utils/random.h"

// A struct to hold all the dimension and configuration parameters for the CosSim layer.
struct CosSimLayerParams {
    int B;       // Batch size
    int in_dim;  // Input dimension
};

template <typename T, int BW, int smallBW, int F, int K_INT>
struct CosSimRandomness {
    SquareRandomness<T, BW, smallBW, F, K_INT, 3> square_randomness;
    ZeroExtendRandomness<T, BW, smallBW, F, K_INT, 2> zero_extend_randomness_square;
    InvSqrtRandomness<T, BW, smallBW, F, K_INT, 2> inv_sqrt_randomness;
    ElementwiseMulRandomness<T, BW, smallBW, F, K_INT, 3> elementwise_mul_randomness;
    ZeroExtendRandomness<T, BW, smallBW, F, K_INT, 3> zero_extend_randomness_elemul;

    MatmulRandomness<T, BW, F, K_INT, 2, 2, 2> matmul_randomness_cosine;
    ZeroExtendRandomness<T, BW, smallBW, F, K_INT, 2> zero_extend_randomness_cosine;

    ZeroExtendRandomness<T, BW, smallBW, F, K_INT, 3> zero_extend_randomness_bwd_all;
    MatmulRandomness<T, BW, F, K_INT, 2, 2, 2> matmul_randomness_bwd_I;
    MatmulRandomness<T, BW, F, K_INT, 2, 2, 2> matmul_randomness_bwd_T;
    ElementwiseMulRandomness<T, BW, smallBW, F, K_INT, 3> elementwise_mul_randomness_bwd;
    ElementwiseMulRandomness<T, BW, smallBW, F, K_INT, 3> elementwise_mul_randomness_bwd_proj;
    ElementwiseMulRandomness<T, BW, smallBW, F, K_INT, 3> elementwise_mul_randomness_bwd_d;
    ZeroExtendRandomness<T, BW, smallBW, F, K_INT, 3> zero_extend_randomness_bwd_d;
};

// IN_BW: n-f
// OUT_BW: n
template <typename T, int IN_BW, int OUT_BW, int F, int K_INT>
class CosSimLayer {
public:
    using InputTensor = FixTensor<T, IN_BW, F, K_INT, 2>;
    using OutputTensor = FixTensor<T, OUT_BW, F, K_INT, 2>;
    using IncomingGradTensor = FixTensor<T, OUT_BW, F, K_INT, 2>;
    using OutgoingGradTensor = FixTensor<T, OUT_BW, F, K_INT, 2>;
    // For the norm, which is a vector of scalars (one for each item in batch)
    using NormTensor = FixTensor<T, IN_BW, F, K_INT, 2>;

    CosSimLayerParams p;
    CosSimRandomness<T, OUT_BW, IN_BW, F, K_INT> randomness;
    FixTensor<T, IN_BW, F, K_INT, 3> norm_broadcasted;
    FixTensor<T, IN_BW, F, K_INT, 2> norm_image;
    FixTensor<T, IN_BW, F, K_INT, 2> norm_text;
    FixTensor<T, IN_BW, F, K_INT, 3> all_rec;
    // To store values for the backward pass
    InputTensor image_share;
    InputTensor text_share;
    InputTensor image_rec;
    InputTensor text_rec;
    NormTensor norm_rec; // Store the reconstructed norm of each vector in the batch

public:
    CosSimLayer(const CosSimLayerParams& params) : p(params),
        norm_broadcasted(2, p.B, p.in_dim),
        norm_image(p.B, p.in_dim),
        norm_text(p.B, p.in_dim),
        all_rec(2, p.B, p.in_dim),
        image_share(p.B, p.in_dim),
        text_share(p.B, p.in_dim),
        image_rec(p.B, p.in_dim),
        text_rec(p.B, p.in_dim),
        norm_rec(2, p.B)
    {}

    size_t get_randomness_size() {
        size_t total_size = 0;
        // forward
        total_size += get_square_random_size<T, 3>(2, p.B, p.in_dim);
        total_size += get_zero_extend_random_size<T, 2>(-1, 2, p.B);
        total_size += get_inv_sqrt_random_size<T, 2>(-1, 2, p.B);
        total_size += get_elementwise_mul_random_size<T, 3>(2, p.B, p.in_dim);
        total_size += get_zero_extend_random_size<T, 3>(2, p.B, p.in_dim);
        total_size += get_matmul_random_size<T, OUT_BW, F, K_INT, 2, 2, 2>(p.B, p.in_dim, p.B, -1);
        total_size += get_zero_extend_random_size<T, 2>(-1, p.B, p.B);
        
        // backward

        total_size += get_zero_extend_random_size<T, 3>(2, p.B, p.in_dim);
        total_size += get_matmul_random_size<T, OUT_BW, F, K_INT, 2, 2, 2>(p.B, p.B, p.in_dim, -1);
        total_size += get_matmul_random_size<T, OUT_BW, F, K_INT, 2, 2, 2>(p.B, p.B, p.in_dim, -1);
        total_size += get_elementwise_mul_random_size<T, 3>(2, p.B, p.in_dim);
        total_size += get_elementwise_mul_random_size<T, 3>(2, p.B, p.in_dim);
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
        randomness.matmul_randomness_cosine = read_matmul_randomness<T, OUT_BW, F, K_INT, 2, 2, 2>(mpc, -1, p.B, p.in_dim, p.B);
        randomness.zero_extend_randomness_cosine = read_zero_extend_randomness<T, OUT_BW, IN_BW, F, K_INT, 2>(mpc, 2, p.B, p.B);
        // backward for image
        randomness.zero_extend_randomness_bwd_all = read_zero_extend_randomness<T, OUT_BW, IN_BW, F, K_INT, 3>(mpc, 2, p.B, p.in_dim);
        randomness.matmul_randomness_bwd_I = read_matmul_randomness<T, OUT_BW, F, K_INT, 2, 2, 2>(mpc, -1, p.B, p.B, p.in_dim);
        randomness.matmul_randomness_bwd_T = read_matmul_randomness<T, OUT_BW, F, K_INT, 2, 2, 2>(mpc, -1, p.B, p.B, p.in_dim);
        randomness.elementwise_mul_randomness_bwd = read_elementwise_mul_randomness<T, OUT_BW, IN_BW, F, K_INT, 3>(mpc, 2, p.B, p.in_dim);
        randomness.elementwise_mul_randomness_bwd_proj = read_elementwise_mul_randomness<T, OUT_BW, IN_BW, F, K_INT, 3>(mpc, 2, p.B, p.in_dim);
        randomness.elementwise_mul_randomness_bwd_d = read_elementwise_mul_randomness<T, OUT_BW, IN_BW, F, K_INT, 3>(mpc, 2, p.B, p.in_dim);
        randomness.zero_extend_randomness_bwd_d = read_zero_extend_randomness<T, OUT_BW, IN_BW, F, K_INT, 3>(mpc, 2, p.B, p.in_dim);
    }
    
    
    void generate_randomness(Buffer& p0_buf, Buffer& p1_buf) {
        // generate_square_randomness<T, IN_BW, F, K_INT, 2>(p0_buf, p1_buf);
        FixTensor<T, IN_BW, F, K_INT, 3> R(2, p.B, p.in_dim);
        FixTensor<T, OUT_BW, F, K_INT, 3> R_N(2, p.B, p.in_dim);
        FixTensor<T, OUT_BW, F, K_INT, 3> R_SQUARE(2, p.B, p.in_dim);
        FixTensor<T, OUT_BW, F, K_INT, 3> R_MSB(2, p.B, p.in_dim);
        FixTensor<T, OUT_BW, F, K_INT, 3> R_R_MSB(2, p.B, p.in_dim);
        Random rg;
        T* val = rg.template randomGE<T>(2 * p.B * p.in_dim, IN_BW);
        for(int i = 0; i < 2; i++) {
            for(int j = 0; j < p.B; j++) {
                for(int k = 0; k < p.in_dim; k++) {
                    R(i, j, k) = Fix<T, IN_BW, F, K_INT>(val[i * p.in_dim * p.B + j * p.in_dim + k]);
                    R_N(i, j, k) = Fix<T, OUT_BW, F, K_INT>(val[i * p.in_dim * p.B + j * p.in_dim + k]);
                }
            }
        }
        delete[] val;
        R_SQUARE = R_N * R_N;
        R_MSB = get_msb<OUT_BW, F, K_INT>(R);
        R_R_MSB = R_N * R_MSB;
        secret_share_and_write_tensor(R, p0_buf, p1_buf);
        secret_share_and_write_tensor(R_N, p0_buf, p1_buf);
        secret_share_and_write_tensor(R_SQUARE, p0_buf, p1_buf);
        secret_share_and_write_tensor(R_MSB, p0_buf, p1_buf);
        secret_share_and_write_tensor(R_R_MSB, p0_buf, p1_buf);
        
        generate_zero_extend_randomness<T, OUT_BW, IN_BW, F, K_INT, 2>(p0_buf, p1_buf, -1, 2, p.B);
        generate_inv_sqrt_randomness<T, OUT_BW, IN_BW, F, K_INT, 2>(p0_buf, p1_buf, -1, 2, p.B);
        FixTensor<T, IN_BW, F, K_INT, 3> r_x_m(2, p.B, p.in_dim), r_y_m(2, p.B, p.in_dim);
        r_x_m = R;
        val = rg.template randomGE<T>(2 * p.B, IN_BW);
        for(int i = 0; i < 2; i++) {
            for(int j = 0; j < p.B; j++) {
                for(int k = 0; k < p.in_dim; k++) {
                    r_y_m(i, j, k) = Fix<T, IN_BW, F, K_INT>(val[i * p.B + j]);
                }
            }
        }
        delete[] val;
        generate_elementwise_mul_randomness<T, OUT_BW, IN_BW, F, K_INT, 3, Eigen::RowMajor>(p0_buf, p1_buf, 2, p.B, p.in_dim, &r_x_m, &r_y_m);
        generate_zero_extend_randomness<T, OUT_BW, IN_BW, F, K_INT, 3>(p0_buf, p1_buf, 2, p.B, p.in_dim);
        generate_matmul_randomness<T, OUT_BW, F, K_INT, 2, 2, 2>(p0_buf, p1_buf, -1, p.B, p.in_dim, p.B);
        generate_zero_extend_randomness<T, OUT_BW, IN_BW, F, K_INT, 2>(p0_buf, p1_buf, -1, p.B, p.B);

        // backward randomness
        FixTensor<T, IN_BW, F, K_INT, 2> R_I(p.B, p.in_dim);
        FixTensor<T, IN_BW, F, K_INT, 2> R_T(p.B, p.in_dim);
        FixTensor<T, IN_BW, F, K_INT, 2> R_Norm_I(p.B, p.in_dim);
        FixTensor<T, IN_BW, F, K_INT, 2> R_Norm_T(p.B, p.in_dim);
        for(int i = 0; i < p.B; i++) {
            for(int j = 0; j < p.in_dim; j++) {
                R_I(i, j) = R(0, i, j);
                R_T(i, j) = R(1, i, j);
                R_Norm_I(i, j) = r_y_m(0, i, j);
                R_Norm_T(i, j) = r_y_m(1, i, j);
            }
        }
        generate_zero_extend_randomness<T, OUT_BW, IN_BW, F, K_INT, 3>(p0_buf, p1_buf, 2, p.B, p.in_dim, &R);
        FixTensor<T, OUT_BW, F, K_INT, 2> U_I(p.B, p.B);
        val = rg.template randomGE<T>(p.B * p.B, OUT_BW);
        for(int i = 0; i < p.B; i++) {
            for(int j = 0; j < p.B; j++) {
                U_I(i, j) = Fix<T, OUT_BW, F, K_INT>(val[i * p.B + j]);
            }
        }
        delete[] val;
        // U_I.setRandom();
        // std::cout << "U_I is " << U_I << std::endl;
        generate_matmul_randomness<T, OUT_BW, F, K_INT, 2, 2, 2>(p0_buf, p1_buf, -1, p.B, p.B, p.in_dim, &U_I);
        FixTensor<T, OUT_BW, F, K_INT, 2> U_I_T(p.B, p.B);
        U_I_T = U_I.shuffle(Eigen::array<int, 2>{1, 0});
        generate_matmul_randomness<T, OUT_BW, F, K_INT, 2, 2, 2>(p0_buf, p1_buf, -1, p.B, p.B, p.in_dim, &U_I_T);
        

        generate_elementwise_mul_randomness<T, OUT_BW, IN_BW, F, K_INT, 3, Eigen::RowMajor>(p0_buf, p1_buf, 2, p.B, p.in_dim, nullptr, &R);
        generate_elementwise_mul_randomness<T, OUT_BW, IN_BW, F, K_INT, 3, Eigen::RowMajor>(p0_buf, p1_buf, 2, p.B, p.in_dim, &R, nullptr);
        generate_elementwise_mul_randomness<T, OUT_BW, IN_BW, F, K_INT, 3, Eigen::RowMajor>(p0_buf, p1_buf, 2, p.B, p.in_dim, nullptr, &r_y_m);
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
        FixTensor<T, IN_BW, F, K_INT, 3> all_rec = all_share + randomness.square_randomness.R;
        this->all_rec = reconstruct_tensor(all_rec);
        
        for(int j = 0; j < p.B; j++) {
            for(int k = 0; k < p.in_dim; k++) {
                this->image_rec(j, k) = this->all_rec(0, j, k);
                this->text_rec(j, k) = this->all_rec(1, j, k);
            }
        }
        
        // --- L2 Normalization Forward---
        // y = x / ||x||_2
        // Operations are performed row-wise for the batch.

        // 1. x_squared = x * x (element-wise)
        // TODO: Implement secure_elemwise_mul if not available
        // x_share bw: n - f, x_square bw: 
        FixTensor<T, OUT_BW, F, K_INT, 3> all_square = square_tensor_opt<T, OUT_BW, IN_BW, F, K_INT, 3, Eigen::RowMajor>(this->all_rec, randomness.square_randomness, true);
        FixTensor<T, IN_BW, F, K_INT, 3> all_square_m = truncate_reduce_tensor(all_square);
        // 2. sum_sq = sum(x_squared) row-wise
        // TODO: Implement sum_rows for FixTensor
        // This will result in a tensor of shape (2, B)
        FixTensor<T, IN_BW, F, K_INT, 2> all_sum_sq_m = sum_reduce_tensor<2, T, IN_BW, F, K_INT, Eigen::RowMajor>(all_square_m);
        FixTensor<T, OUT_BW, F, K_INT, 2> all_sum_sq = zero_extend_tensor<T, OUT_BW, IN_BW, F, K_INT, 2>(all_sum_sq_m, randomness.zero_extend_randomness_square);
        // 3. norm = 1/sqrt(sum_sq)
        // TODO: Implement secure_sqrt for MPC
        // This is a complex protocol. For now, we can use a placeholder.
        // shape: (2, B)
        FixTensor<T, IN_BW, F, K_INT, 2> norm = inv_sqrt_tensor<T, OUT_BW, IN_BW, F, K_INT, 2>(all_sum_sq, randomness.inv_sqrt_randomness);
        FixTensor<T, IN_BW, F, K_INT, 2> r_y_m_2d(2, p.B);
        for (int i = 0; i < 2; i++) {
            for (int j = 0; j < p.B; j++) {
                r_y_m_2d(i, j) = randomness.elementwise_mul_randomness.r_y_m(i, j, 0);
            }
        }
        
        norm = norm + r_y_m_2d;
        this->norm_rec = reconstruct_tensor(norm);
        // 4. y = x * 1 / norm (element-wise)
        // Need to broadcast norm_rec to match shape of x_share
        
        for (int i = 0; i < 2; i++) {
            for (int j = 0; j < p.B; j++) {
                for (int k = 0; k < p.in_dim; k++) {
                    this->norm_broadcasted(i, j, k) = this->norm_rec(i, j);
                    if (i == 0) {
                        this->norm_image(j, k) = this->norm_rec(i, j);
                    }
                    else {
                        this->norm_text(j, k) = this->norm_rec(i, j);
                    }
                }
            }
        }
        
        FixTensor<T, OUT_BW, F, K_INT, 3> y = elementwise_mul_opt<T, OUT_BW, IN_BW, F, K_INT, 3>(this->all_rec, norm_broadcasted, randomness.elementwise_mul_randomness, true, true);
        FixTensor<T, IN_BW, F, K_INT, 3> y_m = truncate_reduce_tensor(y);
        y = zero_extend_tensor<T, OUT_BW, IN_BW, F, K_INT, 3>(y_m, randomness.zero_extend_randomness_elemul);
        FixTensor<T, OUT_BW, F, K_INT, 2> y_I(p.B, p.in_dim);
        FixTensor<T, OUT_BW, F, K_INT, 2> y_T_T(p.in_dim, p.B);
        for (int i = 0; i < p.B; i++) {
            for (int j = 0; j < p.in_dim; j++) {
                y_I(i, j) = y(0, i, j);
                y_T_T(j, i) = y(1, i, j);
            }
        }
        FixTensor<T, OUT_BW, F, K_INT, 2> similarity = secure_matmul(y_I, y_T_T, randomness.matmul_randomness_cosine);
        FixTensor<T, IN_BW, F, K_INT, 2> similarity_m = truncate_reduce_tensor(similarity);
        similarity = zero_extend_tensor<T, OUT_BW, IN_BW, F, K_INT, 2>(similarity_m, randomness.zero_extend_randomness_cosine);
        return similarity;
    }

    std::pair<FixTensor<T, OUT_BW, F, K_INT, 2>, FixTensor<T, OUT_BW, F, K_INT, 2>> backward(const IncomingGradTensor& incoming_grad_share) {
        if (mpc_instance == nullptr) throw std::runtime_error("MPC instance must be initialized before using backward.");
        
        // extend the all_rec tensor to OUT_BW
        FixTensor<T, OUT_BW, F, K_INT, 3> all_ext_share = zero_extend_tensor<T, OUT_BW, IN_BW, F, K_INT, 3>(this->all_rec, randomness.zero_extend_randomness_bwd_all, true);
        
        FixTensor<T, OUT_BW, F, K_INT, 2> image_ext_share(p.B, p.in_dim);
        FixTensor<T, OUT_BW, F, K_INT, 2> text_ext_share(p.B, p.in_dim);
        for(int i = 0; i < p.B; i++) {
            for(int j = 0; j < p.in_dim; j++) {
                image_ext_share(i, j) = all_ext_share(0, i, j);
                text_ext_share(i, j) = all_ext_share(1, i, j);
            }
        }

        // --- L2 Normalization Backward ---
        // for image
        // dI = (dL/dy) @ image  (B, B) @ (B, dim) -> (B, dim)
        FixTensor<T, OUT_BW, F, K_INT, 2> incoming_grad_rec = incoming_grad_share - randomness.matmul_randomness_bwd_I.U;
        FixTensor<T, OUT_BW, F, K_INT, 2> image_ext_rec = image_ext_share - randomness.matmul_randomness_bwd_I.V;
        FixTensor<T, OUT_BW, F, K_INT, 2> text_ext_rec = text_ext_share - randomness.matmul_randomness_bwd_T.V;
        reconstruct_tensor_parallel(incoming_grad_rec, image_ext_rec, text_ext_rec);
        FixTensor<T, OUT_BW, F, K_INT, 2> incoming_grad_share_T = incoming_grad_share.shuffle(Eigen::array<int, 2>{1, 0});
        FixTensor<T, OUT_BW, F, K_INT, 2> incoming_grad_rec_T = incoming_grad_rec.shuffle(Eigen::array<int, 2>{1, 0});
        FixTensor<T, OUT_BW, F, K_INT, 2> dI = secure_matmul(incoming_grad_share, image_ext_share, randomness.matmul_randomness_bwd_I, &incoming_grad_rec, &image_ext_rec);
        FixTensor<T, IN_BW, F, K_INT, 2> dI_m = truncate_reduce_tensor(dI);
                
        FixTensor<T, OUT_BW, F, K_INT, 2> dT = secure_matmul(incoming_grad_share_T, text_ext_share, randomness.matmul_randomness_bwd_T, &incoming_grad_rec_T, &text_ext_rec);
        FixTensor<T, IN_BW, F, K_INT, 2> dT_m = truncate_reduce_tensor(dT);
        
        FixTensor<T, IN_BW, F, K_INT, 3> dIT_m(2, p.B, p.in_dim);
        for(int i = 0; i < p.B; i++) {
            for(int j = 0; j < p.in_dim; j++) {
                dIT_m(0, i, j) = dI_m(i, j);
                dIT_m(1, i, j) = dT_m(i, j);
            }
        }
        // proj_I = dI * I
        FixTensor<T, OUT_BW, F, K_INT, 3> Proj = elementwise_mul_opt<T, OUT_BW, IN_BW, F, K_INT, 3>(dIT_m, this->all_rec, randomness.elementwise_mul_randomness_bwd, false, true);
        // proj_I_m = sum(proj_I)
        FixTensor<T, IN_BW, F, K_INT, 3> Proj_tr = truncate_reduce_tensor(Proj);
        FixTensor<T, IN_BW, F, K_INT, 2> Proj_m = sum_reduce_tensor<2, T, IN_BW, F, K_INT, Eigen::RowMajor>(Proj_tr);

        
        FixTensor<T, IN_BW, F, K_INT, 3> Proj_broadcasted(2, p.B, p.in_dim);
        for(int i = 0; i < p.B; i++) {
            for(int j = 0; j < p.in_dim; j++) {
                Proj_broadcasted(0, i, j) = Proj_m(0, i);
                Proj_broadcasted(1, i, j) = Proj_m(1, i);
            }
        }
        // term = proj_I_m * image
        auto term = elementwise_mul_opt<T, OUT_BW, IN_BW, F, K_INT, 3>(this->all_rec, Proj_broadcasted, randomness.elementwise_mul_randomness_bwd_proj, true, false);
        auto term_m = truncate_reduce_tensor(term);

        dIT_m = dIT_m  - term_m;
        auto dIT = elementwise_mul_opt<T, OUT_BW, IN_BW, F, K_INT, 3>(dIT_m, this->norm_broadcasted, randomness.elementwise_mul_randomness_bwd_d, false, true);
        dIT_m = truncate_reduce_tensor(dIT);
        dIT = zero_extend_tensor<T, OUT_BW, IN_BW, F, K_INT, 3>(dIT_m, randomness.zero_extend_randomness_bwd_d);
        for(int i = 0; i < p.B; i++) {  
            for(int j = 0; j < p.in_dim; j++) {
                dI(i, j) = dIT(0, i, j);
                dT(i, j) = dIT(1, i, j);
            }
        }
        
        return std::make_pair(dI, dT);
    }
};
