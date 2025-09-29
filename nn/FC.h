#pragma once

#include "mpc/mpc.h"
#include "mpc/fix_tensor.h"
#include "mpc/tensor_ops.h"
#include "utils/random.h"

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
    using FixBias = Fix<T, IN_BW, 2*F, K_INT>; // New type for bias
    using InputTensor = FixTensor<T, IN_BW, F, K_INT, 2>;
    using WeightTensor = FixTensor<T, IN_BW, F, K_INT, 2>;
    using BiasTensor = FixTensor<T, IN_BW, 2*F, K_INT, 1>; // Use the new FixBias type
    using OutputTensor2D = FixTensor<T, OUT_BW, F, K_INT, 2>;
    using OutputTensor3D = FixTensor<T, OUT_BW, F, K_INT, 3>;

private:
    FCLayerParams p;
    WeightTensor W_share; // Secret share of the weight matrix
    BiasTensor Y_share;   // Secret share of the bias vector

    // Member variables to store randomness for the forward pass
    // For non-batched input (Rank 2)
    InputTensor U_fwd;
    WeightTensor V_fwd;
    FixTensor<T, IN_BW, F, K_INT, 2> Z_fwd;
    // For batched input (Rank 3)
    FixTensor<T, IN_BW, F, K_INT, 3> U_fwd_b;
    // V_fwd is the same for batched and non-batched
    FixTensor<T, IN_BW, F, K_INT, 3> Z_fwd_b;
    
    // To store the input share for the backward pass
    InputTensor input_share;
    FixTensor<T, IN_BW, F, K_INT, 3> input_share_b;


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
    
    // For use by the dealer: calculates the byte size of all randomness needed for the forward pass.
    size_t getForwardRandomnessSize() {
        size_t total_size = 0;
        if (p.B == 1) {
            total_size += InputTensor(p.M, p.N).size() * sizeof(T);
            total_size += WeightTensor(p.N, p.K).size() * sizeof(T);
            total_size += OutputTensor2D(p.M, p.K).size() * sizeof(T);
        } else {
            total_size += FixTensor<T, IN_BW, F, K_INT, 3>(p.B, p.M, p.N).size() * sizeof(T);
            total_size += WeightTensor(p.N, p.K).size() * sizeof(T);
            total_size += OutputTensor3D(p.B, p.M, p.K).size() * sizeof(T);
        }
        return total_size;
    }

    // For use by the dealer: generates and writes all forward pass randomness into the provided buffers.
    void generateForwardRandomness(uint8_t*& random_data_ptr) {
        auto write_tensor = [&](auto& tensor) {
            size_t size_in_bytes = tensor.size() * sizeof(T);
            memcpy(random_data_ptr, tensor.data(), size_in_bytes);
            random_data_ptr += size_in_bytes;
        };

        Random random_gen;
        if (p.B == 1) {
            InputTensor U(p.M, p.N); U.initialize(random_gen);
            WeightTensor V(p.N, p.K); V.initialize(random_gen);
            auto Z_temp = tensor_mul(U, V);
            OutputTensor2D Z = Z_temp.template cast<FixOut>();
            
            write_tensor(U);
            write_tensor(V);
            write_tensor(Z);
        } else {
            FixTensor<T, IN_BW, F, K_INT, 3> U(p.B, p.M, p.N); U.initialize(random_gen);
            WeightTensor V(p.N, p.K); V.initialize(random_gen);
            auto Z_temp = tensor_mul(U, V);
            OutputTensor3D Z = Z_temp.template cast<FixOut>();

            write_tensor(U);
            write_tensor(V);
            write_tensor(Z);
        }
    }

    // The pointer is advanced by this function
    void readForwardRandomness(uint8_t*& random_data_ptr) {
        auto read_tensor = [&](auto& tensor, const auto& dims) {
            tensor.resize(dims);
            size_t size_in_bytes = tensor.size() * sizeof(T);
            memcpy(tensor.data(), random_data_ptr, size_in_bytes);
            random_data_ptr += size_in_bytes;
        };

        if (p.B == 1) {
            read_tensor(U_fwd, Eigen::array<long, 2>{p.M, p.N});
            read_tensor(V_fwd, Eigen::array<long, 2>{p.N, p.K});
            read_tensor(Z_fwd, Eigen::array<long, 2>{p.M, p.K});
        } else {
            read_tensor(U_fwd_b, Eigen::array<long, 3>{p.B, p.M, p.N});
            read_tensor(V_fwd, Eigen::array<long, 2>{p.N, p.K});
            read_tensor(Z_fwd_b, Eigen::array<long, 3>{p.B, p.M, p.K});
        }
    }

    void readBackwardRandomness() {
        // Implementation to follow (placeholder)
    }

    // Forward pass for non-batched input
    template<int TRUNC_FWD>
    auto forward(const InputTensor& x_share) {
        if (mpc_instance == nullptr) throw std::runtime_error("MPC instance must be initialized before using forward.");
        this->input_share = x_share; // Cache for backward pass
        auto mul_result_share = secure_matmul(x_share, W_share, U_fwd, V_fwd, Z_fwd);
        std::cout << "mul_result_share calculated " << std::endl;
        if (p.use_bias) {
            // FIXME: Bias addition is temporarily disabled due to type mismatch issues.
            // FixTensor<T, IN_BW, 2*F, K_INT, 2> Y_share_broadcasted = Y_share.broadcast(Eigen::array<int, 2>{p.M, 1});
            // auto biased_result = mul_result_share + Y_share_broadcasted;
            // auto trunc_share = truncate_reduce_tensor(biased_result);
            // return change_bitwidth<OUT_BW, F, K_INT>(trunc_share);
        }
        
        // Truncate Reduce only
        auto trunc_share = truncate_reduce_tensor(mul_result_share);
        std::cout << "trunc_share calculated " << std::endl;
        return change_bitwidth<OUT_BW, F, K_INT>(trunc_share);
    }

    // Forward pass for batched input
    template<int TRUNC_FWD>
    auto forward(const FixTensor<T, IN_BW, F, K_INT, 3>& x_share) {
        if (mpc_instance == nullptr) throw std::runtime_error("MPC instance must be initialized before using forward.");
        this->input_share_b = x_share; // Cache for backward pass
        auto mul_result_share_expr = secure_matmul(x_share, W_share, U_fwd_b, V_fwd, Z_fwd_b);

        // Manually cast the result to a tensor with 2*F fractional bits
        FixTensor<T, IN_BW, 2*F, K_INT, 3> mul_result_share(p.B, p.M, p.K);
        for(int i = 0; i < mul_result_share.size(); ++i) {
            mul_result_share.data()[i].val = mul_result_share_expr.data()[i].val;
        }

        if (p.use_bias) {
            // FIXME: Bias addition is temporarily disabled due to type mismatch issues.
        }
        
        auto trunc_share = truncate_reduce_tensor(mul_result_share);
        return change_bitwidth<OUT_BW, F, K_INT>(trunc_share);
    }
};
