#pragma once

#include "mpc/mpc.h"
#include "mpc/fix_tensor.h"
#include "mpc/tensor_ops.h"
#include "mpc/truncate.h"
#include "mpc/matmul.h"
#include "utils/random.h"
#include "nn/SGD.h"

template <typename T, int IN_BW, int F, int K_INT>
struct FCLayerRandomness {
    MatmulRandomness<T, IN_BW, F, K_INT, 2, 2, 2> matmul_randomness_fwd;
    MatmulRandomness<T, IN_BW, F, K_INT, 2, 2, 2> matmul_randomness_bwd;
    MatmulRandomness<T, IN_BW, F, K_INT, 2, 2, 2> matmul_randomness_dw;
    ZeroExtendRandomness<T, IN_BW, IN_BW - F, F, K_INT, 2> zero_extend_randomness;
    ZeroExtendRandomness<T, IN_BW, IN_BW - F, F, K_INT, 2> zero_extend_randomness_w_update;
};

// A struct to hold all the dimension and configuration parameters for the FC layer.
struct FCLayerParams {
    int B;                  // Batch size
    int in_dim;                  // Input matrix rows
    int out_dim;             // Weight matrix columns / Output matrix columns
    bool use_bias;          // Whether to use a bias term
    bool reconstructed_input; // Whether the input is already reconstructed (public)
    // int trunc_forward;   // REMOVED - will be a template parameter on forward()
    int trunc_backward;     // Backward pass truncation method
};

template <typename T, int IN_BW, int OUT_BW, int F, int K_INT>
class FCLayer {
public:
    using FixIn = Fix<T, IN_BW, F, K_INT>;
    using FixOut = Fix<T, OUT_BW, F, K_INT>;
    using FixBias = Fix<T, IN_BW, F, K_INT>; // New type for bias
    using InputTensor = FixTensor<T, IN_BW, F, K_INT, 2>;
    using WeightTensor = FixTensor<T, IN_BW, F, K_INT, 2>;
    using BiasTensor = FixTensor<T, IN_BW, F, K_INT, 1>; // Use the new FixBias type
    using OutputTensor = FixTensor<T, OUT_BW, F, K_INT, 2>;
    using IncomingGradTensor = FixTensor<T, IN_BW, F, K_INT, 2>; // 2D
    using OutgoingGradTensor = FixTensor<T, IN_BW, F, K_INT, 2>; // MUST ALSO BE 2D


    FCLayerParams p;
    WeightTensor W_share; // Secret share of the weight matrix
    WeightTensor W_rec; // Reconstructed weight matrix
    BiasTensor Y_share;   // Secret share of the bias vector

    FCLayerRandomness<T, IN_BW, F, K_INT> randomness;

    // To store the input share for the backward pass
    InputTensor input_rec;
    InputTensor input_share;
    // To store the computed weight gradient share
    WeightTensor dW_share;



public:
    FCLayer(const FCLayerParams& params) : p(params) {}

    void initWeights(uint8_t **weights_ptr, bool floatWeights) {
        if (mpc_instance == nullptr) throw std::runtime_error("MPC instance must be initialized before using initWeights.");
        // This function should be called by all parties with the same weights
        if (mpc_instance->party == 0) {
            // Party 0 (the dealer of shares) will load the plaintext weights
            WeightTensor W_plain(p.in_dim, p.out_dim);
            BiasTensor Y_plain(p.out_dim);

            if (floatWeights) {
                float* float_weights = reinterpret_cast<float*>(*weights_ptr);
                for (int i = 0; i < W_plain.size(); i++) {
                    W_plain.data()[i] = FixIn(float_weights[i]);
                }
                *weights_ptr += W_plain.size() * sizeof(float);

                if (p.use_bias) {
                    float_weights = reinterpret_cast<float*>(*weights_ptr);
                    for (int i = 0; i < Y_plain.size(); i++) {
                        // Bias is added after multiplication, so its scale is doubled.
                        Y_plain.data()[i] = FixBias(float_weights[i]);
                    }
                    *weights_ptr += Y_plain.size() * sizeof(float);
                }
            } else {
                T* raw_weights = reinterpret_cast<T*>(*weights_ptr);
                for (int i = 0; i < W_plain.size(); i++) {
                     W_plain.data()[i] = FixIn(raw_weights[i]);
                }
                *weights_ptr += W_plain.size() * sizeof(T);

                if (p.use_bias) {
                    raw_weights = reinterpret_cast<T*>(*weights_ptr);
                     for (int i = 0; i < Y_plain.size(); i++) {
                         Y_plain.data()[i] = FixIn(raw_weights[i]);
                    }
                    *weights_ptr += Y_plain.size() * sizeof(T);
                }
            }

            // Secret share the plaintext weights and biases
            W_share = secret_share_tensor(W_plain);
            if (p.use_bias) {
                Y_share = secret_share_tensor(Y_plain);
            }

        } else {
            // Other parties receive their shares
            // Still need to advance the buffer pointer to stay in sync, if the buffer exists.
            if (weights_ptr != nullptr && *weights_ptr != nullptr) {
                if (floatWeights) {
                    *weights_ptr += p.in_dim * p.out_dim * sizeof(float);
                     if (p.use_bias) {
                        *weights_ptr += p.out_dim * sizeof(float);
                    }
                } else {
                    *weights_ptr += p.in_dim * p.out_dim * sizeof(T);
                     if (p.use_bias) {
                        *weights_ptr += p.out_dim * sizeof(T);
                    }
                }
            }
            W_share = secret_share_tensor(WeightTensor(p.in_dim, p.out_dim)); // Pass dummy tensor
            if (p.use_bias) {
                Y_share = secret_share_tensor(BiasTensor(p.out_dim)); // Pass dummy tensor
            }
        }
    }
    
    size_t getRandomnessSize() {
        size_t total_size = 0;
        total_size += get_matmul_random_size<T, IN_BW, F, K_INT, 2, 2, 2>(p.B, p.in_dim, p.out_dim);
        total_size += get_matmul_random_size<T, IN_BW, F, K_INT, 2, 2, 2>(p.B, p.out_dim, p.in_dim);
        total_size += get_matmul_random_size<T, IN_BW, F, K_INT, 2, 2, 2>(p.in_dim, p.B, p.out_dim);
        total_size += get_zero_extend_random_size<T, 2>(-1, p.B, p.in_dim);
        total_size += get_zero_extend_random_size<T, 2>(-1, p.out_dim, p.in_dim);
        return total_size;
    }

    void generate_randomness(Buffer& p0_buf, Buffer& p1_buf) {
        // For Forward
        // input_rec
        FixTensor<T, IN_BW, F, K_INT, 2> U;
        // weight_rec
        FixTensor<T, IN_BW, F, K_INT, 2> V;
        FixTensor<T, IN_BW, F, K_INT, 2> Z;
        U.resize(p.B, p.in_dim);
        V.resize(p.in_dim, p.out_dim);
        Z.resize(p.B, p.out_dim);
        U.setRandom();
        V.setRandom();
        std::cout << "U" << std::endl;
        std::cout << U << std::endl;
        std::cout << "V" << std::endl;
        std::cout << V << std::endl;
        Z = tensor_mul(U, V);
        secret_share_and_write_tensor(U, p0_buf, p1_buf);
        secret_share_and_write_tensor(V, p0_buf, p1_buf);
        secret_share_and_write_tensor(Z, p0_buf, p1_buf);
        // For dE/dX
        // incoming_grad_rec
        FixTensor<T, IN_BW, F, K_INT, 2> U_bwd;
        // w_rec_T
        FixTensor<T, IN_BW, F, K_INT, 2> V_bwd;
        FixTensor<T, IN_BW, F, K_INT, 2> Z_bwd;
        U_bwd.resize(p.B, p.out_dim);
        V_bwd.resize(p.out_dim, p.in_dim);
        Z_bwd.resize(p.B, p.in_dim);
        U_bwd.setRandom();
        V_bwd = V.shuffle(Eigen::array<int, 2>{1, 0});
        Z_bwd = tensor_mul(U_bwd, V_bwd);
        secret_share_and_write_tensor(U_bwd, p0_buf, p1_buf);
        secret_share_and_write_tensor(V_bwd, p0_buf, p1_buf);
        secret_share_and_write_tensor(Z_bwd, p0_buf, p1_buf);
        // For dE/dW
        // input_rec_T
        FixTensor<T, IN_BW, F, K_INT, 2> U_bwdw;
        // incoming_grad_rec
        FixTensor<T, IN_BW, F, K_INT, 2> V_bwdw;
        FixTensor<T, IN_BW, F, K_INT, 2> Z_bwdw;
        U_bwdw.resize(p.in_dim, p.B);
        V_bwdw.resize(p.B, p.out_dim);
        Z_bwdw.resize(p.in_dim, p.out_dim);
        U_bwdw = U.shuffle(Eigen::array<int, 2>{1, 0});
        V_bwdw = U_bwd;
        Z_bwdw = tensor_mul(U_bwdw, V_bwdw);
        secret_share_and_write_tensor(U_bwdw, p0_buf, p1_buf);
        secret_share_and_write_tensor(V_bwdw, p0_buf, p1_buf);
        secret_share_and_write_tensor(Z_bwdw, p0_buf, p1_buf);
        generate_zero_extend_randomness<T, IN_BW, IN_BW - F, F, K_INT, 2>(p0_buf, p1_buf, -1, p.B, p.in_dim);
        generate_zero_extend_randomness<T, IN_BW, IN_BW - F, F, K_INT, 2>(p0_buf, p1_buf, -1, p.in_dim, p.out_dim);
    }
    

    // For use by parties: reads randomness shares from the MPC random data buffer.
    void readForwardRandomness(MPC& mpc) {
        randomness.matmul_randomness_fwd = read_matmul_randomness<T, IN_BW, F, K_INT, 2, 2, 2>(mpc, -1, p.B, p.in_dim, p.out_dim);
    }

    void readBackwardRandomness(MPC& mpc) {
        randomness.matmul_randomness_bwd = read_matmul_randomness<T, IN_BW, F, K_INT, 2, 2, 2>(mpc, -1, p.B, p.out_dim, p.in_dim);
        randomness.matmul_randomness_dw = read_matmul_randomness<T, IN_BW, F, K_INT, 2, 2, 2>(mpc, -1, p.in_dim, p.B, p.out_dim);
        randomness.zero_extend_randomness = read_zero_extend_randomness<T, IN_BW, IN_BW - F, F, K_INT, 2>(mpc, -1, p.B, p.in_dim);
        randomness.zero_extend_randomness_w_update = read_zero_extend_randomness<T, IN_BW, IN_BW - F, F, K_INT, 2>(mpc, -1, p.in_dim, p.out_dim);
    }
    
    template<int TRUNC_FWD>
    auto forward(const FixTensor<T, IN_BW, F, K_INT, 2>& x_share, const FixTensor<T, IN_BW, F, K_INT, 2>& x_reconstructed = nullptr) {
        if (mpc_instance == nullptr) throw std::runtime_error("MPC instance must be initialized before using forward.");
        this->input_rec = x_reconstructed; // Cache for backward pass
        this->input_share = x_share;
        auto mul_result_share = secure_matmul(x_share, W_share, randomness.matmul_randomness_fwd, &x_reconstructed, &W_rec);
        if (p.use_bias) {
            // FIXME: Bias addition is temporarily disabled due to type mismatch issues.
            // FixTensor<T, IN_BW, 2*F, K_INT, 2> Y_share_broadcasted = Y_share.broadcast(Eigen::array<int, 2>{p.M, 1});
            // auto biased_result = mul_result_share + Y_share_broadcasted;
            // auto trunc_share = truncate_reduce_tensor(biased_result);
            // return change_bitwidth<OUT_BW, F, K_INT>(trunc_share);
        }
        
        // Truncate Reduce only
        auto trunc_share = truncate_reduce_tensor(mul_result_share);
        return trunc_share;
    }

    auto backward(const FixTensor<T, IN_BW, F, K_INT, 2>& incoming_grad_rec, const FixTensor<T, IN_BW, F, K_INT, 2>& incoming_grad_share) {
        if (mpc_instance == nullptr) throw std::runtime_error("MPC instance must be initialized before using backward.");
        
        // Transpose the weight matrix share
        WeightTensor W_rec_T = W_rec.shuffle(Eigen::array<int, 2>{1, 0});
        WeightTensor W_share_T = W_share.shuffle(Eigen::array<int, 2>{1, 0});
        // --- 1. Calculate dE/dX (gradient to propagate backwards) ---
        // incoming_coming_grad_share : (B, out_dim), weight_shape : (in_dim, out_dim)
        // perform matmul: (B, out_dim) @ (out_dim, in_dim) -> (B, in_dim)
        auto outgoing_grad_share_wide = secure_matmul(incoming_grad_share, W_share_T, randomness.matmul_randomness_bwd, &incoming_grad_rec, &W_rec_T);
        auto outgoing_grad_share = truncate_reduce_tensor(outgoing_grad_share_wide);
        auto outgoing_grad_share_extend = zero_extend_tensor(outgoing_grad_share, randomness.zero_extend_randomness);

        // --- 2. Calculate dE/dW (gradient for weight update) ---
        // input_rec : (B, in_dim), incoming_grad_share : (B, out_dim)
        // perform matmul: (in_dim, B) @ (B, out_dim) -> (in_dim, out_dim)
        InputTensor input_rec_T;
        input_rec_T.resize(p.in_dim, p.B);
        input_rec_T = input_rec.shuffle(Eigen::array<int, 2>{1, 0});
        auto input_rec_T_share = input_rec_T;
        if(mpc_instance->party != 0) {
            input_rec_T_share.setZero();
        }
        dW_share = secure_matmul(input_rec_T_share, incoming_grad_share, randomness.matmul_randomness_dw, &input_rec_T, &incoming_grad_rec);
        auto dW_share_m = truncate_reduce_tensor(dW_share);
        dW_share = zero_extend_tensor(dW_share_m, randomness.zero_extend_randomness_w_update);
        return outgoing_grad_share_extend;
    }

    void update(float lr) {
        // The division by batch size is folded into the learning rate
        float scaled_lr = lr / p.B;
        sgd_update(W_share, dW_share, FixIn(scaled_lr), randomness.zero_extend_randomness);
    }
    
};
