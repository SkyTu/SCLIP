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
    using FixBias = Fix<T, IN_BW, F, K_INT>; // New type for bias
    using InputTensor = FixTensor<T, IN_BW, F, K_INT, 3>;
    using WeightTensor = FixTensor<T, IN_BW, F, K_INT, 2>;
    using BiasTensor = FixTensor<T, IN_BW, F, K_INT, 1>; // Use the new FixBias type
    using OutputTensor3D = FixTensor<T, OUT_BW, F, K_INT, 3>;
    using IncomingGradTensor = FixTensor<T, IN_BW, F, K_INT, 3>;
    using OutgoingGradTensor = FixTensor<T, IN_BW, F, K_INT, 3>;

private:
    FCLayerParams p;
    WeightTensor W_share; // Secret share of the weight matrix
    BiasTensor Y_share;   // Secret share of the bias vector

    // Member variables to store randomness for the forward pass
    FixTensor<T, IN_BW, F, K_INT, 3> U_fwd;
    FixTensor<T, IN_BW, F, K_INT, 2> V_fwd;
    FixTensor<T, IN_BW, F, K_INT, 3> Z_fwd;
    
    // Member variables to store randomness for the backward pass
    IncomingGradTensor U_bwd;
    WeightTensor V_bwd; // This is W^T, so its dimensions are KxN
    OutgoingGradTensor Z_bwd;

    // To store the input share for the backward pass
    InputTensor input_share;


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
        total_size += FixTensor<T, IN_BW, F, K_INT, 3>(p.B, p.M, p.N).size() * sizeof(T);
        total_size += WeightTensor(p.N, p.K).size() * sizeof(T);
        total_size += OutputTensor3D(p.B, p.M, p.K).size() * sizeof(T);
        return total_size;
    }

    // For use by the dealer: generates plaintext U,V,Z, secret shares them, and writes the shares to the provided buffers.
    void dealer_generate_randomness(uint8_t*& p0_buf, uint8_t*& p1_buf) {
        // 1. Generate plaintext U, V, Z
        FixTensor<T, IN_BW, F, K_INT, 3> U(p.B, p.M, p.N); U.initialize();
        WeightTensor V(p.N, p.K); V.initialize();
        auto Z = tensor_mul(U, V);

        // 2. Secret share and write each tensor to the buffers
        secret_share_and_write_tensor(U, p0_buf, p1_buf);
        secret_share_and_write_tensor(V, p0_buf, p1_buf);
        secret_share_and_write_tensor(Z, p0_buf, p1_buf);
    }

    // For use by parties: reads randomness shares from the MPC random data buffer.
    void readForwardRandomness(MPC& mpc) {
        uint8_t* random_data_ptr = mpc.random_data.data() + mpc.random_data_idx;
        
        auto read_tensor = [&](auto& tensor, const auto& dims) {
            tensor.resize(dims);
            size_t num_elements = tensor.size();
            size_t size_in_bytes = num_elements * sizeof(T);

            T* src_ptr = reinterpret_cast<T*>(random_data_ptr);
            for (size_t i = 0; i < num_elements; ++i) {
                tensor.data()[i] = typename std::decay_t<decltype(tensor)>::Scalar(src_ptr[i]);
            }
            random_data_ptr += size_in_bytes;
        };

        read_tensor(U_fwd, Eigen::array<long, 3>{p.B, p.M, p.N});
        read_tensor(V_fwd, Eigen::array<long, 2>{p.N, p.K});
        read_tensor(Z_fwd, Eigen::array<long, 3>{p.B, p.M, p.K});

        mpc.random_data_idx = random_data_ptr - mpc.random_data.data();
    }

    // --- Backward Pass Randomness ---
    size_t getBackwardRandomnessSize() {
        size_t total_size = 0;
        total_size += IncomingGradTensor(p.B, p.M, p.K).size() * sizeof(T); // U_bwd
        total_size += WeightTensor(p.K, p.N).size() * sizeof(T);         // V_bwd (this is W^T)
        total_size += OutgoingGradTensor(p.B, p.M, p.N).size() * sizeof(T); // Z_bwd
        return total_size;
    }

    void dealer_generate_backward_randomness(uint8_t*& p0_buf, uint8_t*& p1_buf) {
        IncomingGradTensor U(p.B, p.M, p.K); U.initialize();
        WeightTensor V(p.K, p.N); V.initialize(); // V is KxN

        auto Z = tensor_mul(U, V);
        secret_share_and_write_tensor(U, p0_buf, p1_buf);
        secret_share_and_write_tensor(V, p0_buf, p1_buf);
        secret_share_and_write_tensor(Z, p0_buf, p1_buf);
    }
    
    void readBackwardRandomness(MPC& mpc) {
        uint8_t* random_data_ptr = mpc.random_data.data() + mpc.random_data_idx;
        
        auto read_tensor = [&](auto& tensor, const auto& dims) {
            tensor.resize(dims);
            size_t num_elements = tensor.size();
            size_t size_in_bytes = num_elements * sizeof(T);

            T* src_ptr = reinterpret_cast<T*>(random_data_ptr);
            for (size_t i = 0; i < num_elements; ++i) {
                tensor.data()[i] = typename std::decay_t<decltype(tensor)>::Scalar(src_ptr[i]);
            }
            random_data_ptr += size_in_bytes;
        };

        read_tensor(U_bwd, Eigen::array<long, 3>{p.B, p.M, p.K});
        read_tensor(V_bwd, Eigen::array<long, 2>{p.K, p.N});
        read_tensor(Z_bwd, Eigen::array<long, 3>{p.B, p.M, p.N});

        mpc.random_data_idx = random_data_ptr - mpc.random_data.data();
    }
    
    // --- Forward and Backward Pass ---

    // Forward pass for non-batched input
    template<int TRUNC_FWD>
    auto forward(const FixTensor<T, IN_BW, F, K_INT, 3>& x_share) {
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

    auto backward(const IncomingGradTensor& incoming_grad_share) {
        if (mpc_instance == nullptr) throw std::runtime_error("MPC instance must be initialized before using backward.");
        
        // Transpose the weight matrix share
        WeightTensor W_share_T = W_share.shuffle(Eigen::array<int, 2>{1, 0});
        
        // Secure MatMul: [dE/dX] = [dE/dY] * [W]^T
        auto outgoing_grad_share_wide = secure_matmul(incoming_grad_share, W_share_T, U_bwd, V_bwd, Z_bwd);

        auto trunc_share = truncate_reduce_tensor(outgoing_grad_share_wide);
        std::cout << "trunc_share calculated " << std::endl;
        // The result of the multiplication has 2*f fractional bits and needs to be truncated.
        return trunc_share;
    }
    
};
