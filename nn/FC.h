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
    MatmulRandomness<T, IN_BW, F, K_INT, 3, 2, 3> matmul_randomness_fwd;
    MatmulRandomness<T, IN_BW, F, K_INT, 2, 2, 2> matmul_randomness_bwd;
    MatmulRandomness<T, IN_BW, F, K_INT, 2, 2, 2> matmul_randomness_dw;
    ZeroExtendRandomness<T, IN_BW, IN_BW - F, F, K_INT, 2> zero_extend_randomness;
};

// A struct to hold all the dimension and configuration parameters for the FC layer.
struct FCLayerParams {
    int B;                  // Batch size
    int M;                  // Input matrix rows
    int N;                  // Input matrix columns / Weight matrix rows
    int K;                  // Weight matrix columns / Output matrix columns
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
    using InputTensor = FixTensor<T, IN_BW, F, K_INT, 3>;
    using WeightTensor = FixTensor<T, IN_BW, F, K_INT, 2>;
    using BiasTensor = FixTensor<T, IN_BW, F, K_INT, 1>; // Use the new FixBias type
    using OutputTensor3D = FixTensor<T, OUT_BW, F, K_INT, 3>;
    using IncomingGradTensor = FixTensor<T, IN_BW, F, K_INT, 2>; // 2D
    using OutgoingGradTensor = FixTensor<T, IN_BW, F, K_INT, 2>; // MUST ALSO BE 2D

    // Type definition for the truncated update term in SGD
    using TruncatedGradTensor = FixTensor<T, IN_BW - F, F, K_INT, 2>;

    FCLayerParams p;
    WeightTensor W_share; // Secret share of the weight matrix
    WeightTensor W_rec; // Reconstructed weight matrix
    BiasTensor Y_share;   // Secret share of the bias vector

    FCLayerRandomness<T, IN_BW, F, K_INT> randomness;

    // To store the input share for the backward pass
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
            WeightTensor W_plain(p.N, p.K);
            BiasTensor Y_plain(p.K);

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
                    *weights_ptr += p.N * p.K * sizeof(float);
                     if (p.use_bias) {
                        *weights_ptr += p.K * sizeof(float);
                    }
                } else {
                    *weights_ptr += p.N * p.K * sizeof(T);
                     if (p.use_bias) {
                        *weights_ptr += p.K * sizeof(T);
                    }
                }
            }
            W_share = secret_share_tensor(WeightTensor(p.N, p.K)); // Pass dummy tensor
            if (p.use_bias) {
                Y_share = secret_share_tensor(BiasTensor(p.K)); // Pass dummy tensor
            }
        }
    }
    
    size_t getRandomnessSize() {
        size_t total_size = 0;
        total_size += get_matmul_random_size<T, IN_BW, F, K_INT, 3, 2, 3>(p.M, p.N, p.K, p.B);
        total_size += get_matmul_random_size<T, IN_BW, F, K_INT, 2, 2, 2>(p.M, p.N, p.K, p.B);
        total_size += get_matmul_random_size<T, IN_BW, F, K_INT, 2, 2, 2>(p.N, p.M, p.K, p.B);
        total_size += get_zero_extend_random_size<T, 2>(p.N, p.K, p.B);
        return total_size;
    }

    void generate_randomness(Buffer& p0_buf, Buffer& p1_buf) {
        // For Forward
        generate_matmul_randomness<T, IN_BW, F, K_INT, 3, 2, 3>(p0_buf, p1_buf, p.B, p.M, p.N, p.K);
        // For dE/dX
        generate_matmul_randomness<T, IN_BW, F, K_INT, 2, 2, 2>(p0_buf, p1_buf, p.B, p.M, p.N, p.K);
        // For dE/dW
        generate_matmul_randomness<T, IN_BW, F, K_INT, 2, 2, 2>(p0_buf, p1_buf, p.B, p.N, p.M, p.K);
        generate_zero_extend_randomness<T, IN_BW, IN_BW - F, F, K_INT, 2>(p0_buf, p1_buf, p.B, p.N, p.K);
    }
    

    // For use by parties: reads randomness shares from the MPC random data buffer.
    void readForwardRandomness(MPC& mpc) {
        randomness.matmul_randomness_fwd = read_matmul_randomness<T, IN_BW, F, K_INT, 3, 2, 3>(mpc, p.B, p.M, p.N, p.K);
    }

    void readBackwardRandomness(MPC& mpc) {
        randomness.matmul_randomness_bwd = read_matmul_randomness<T, IN_BW, F, K_INT, 2, 2, 2>(mpc, p.B, p.M, p.N, p.K);
        randomness.matmul_randomness_dw = read_matmul_randomness<T, IN_BW, F, K_INT, 2, 2, 2>(mpc, p.B, p.N, p.M, p.K);
        randomness.zero_extend_randomness = read_zero_extend_randomness<T, IN_BW, IN_BW - F, F, K_INT, 2>(mpc, p.B, p.N, p.K);
    }
    
    template<int TRUNC_FWD>
    auto forward(const FixTensor<T, IN_BW, F, K_INT, 3>& x_share, const FixTensor<T, IN_BW, F, K_INT, 3>& x_reconstructed = nullptr) {
        if (mpc_instance == nullptr) throw std::runtime_error("MPC instance must be initialized before using forward.");
        this->input_share = x_share; // Cache for backward pass
        for(int i = 0; i < x_reconstructed.size(); ++i) {
            std::cout << "x_reconstructed(" << i << "): " << x_reconstructed.data()[i].val << std::endl;
        }
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

    auto backward(const IncomingGradTensor& incoming_grad_share) {
        if (mpc_instance == nullptr) throw std::runtime_error("MPC instance must be initialized before using backward.");
        
        // Transpose the weight matrix share
        WeightTensor W_rec_T = W_rec.shuffle(Eigen::array<int, 2>{1, 0});
        WeightTensor W_share_T = W_share.shuffle(Eigen::array<int, 2>{1, 0});
        // --- 1. Calculate dE/dX (gradient to propagate backwards) ---
        // All tensors are 2D. No broadcasting needed.
        auto outgoing_grad_share_wide = secure_matmul(incoming_grad_share, W_share_T, randomness.matmul_randomness_bwd, nullptr, &W_rec_T);
        auto outgoing_grad_share = truncate_reduce_tensor(outgoing_grad_share_wide);
        auto outgoing_grad_share_extend = zero_extend_tensor(outgoing_grad_share, randomness.zero_extend_randomness);

        // --- 2. Calculate dE/dW (gradient for weight update) ---
        auto input_share_sum = sum_reduce_tensor(this->input_share);
        auto input_share_sum_T_expr = input_share_sum.shuffle(Eigen::array<int, 2>{1, 0});
        WeightTensor input_share_sum_T = input_share_sum_T_expr;
        
        // incoming_grad_share is already the 2D "average" gradient, no reduction needed
        auto dW_share_wide = secure_matmul(input_share_sum_T, incoming_grad_share, randomness.matmul_randomness_dw);
        dW_share = dW_share_wide; // No truncation here, done in sgd_update

        return outgoing_grad_share_extend;
    }

    void update(float lr) {
        // The division by batch size is folded into the learning rate
        float scaled_lr = lr / p.B;
        sgd_update(W_share, dW_share, FixIn(scaled_lr), randomness.zero_extend_randomness);
    }
    
};
