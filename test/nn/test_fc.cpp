#include "nn/FC.h"
#include "mpc/mpc.h"
#include "mpc/tensor_ops.h"
#include <iostream>
#include <vector>
#include <cassert>

void test_fc_forward(MPC& mpc) {
    std::cout << "--- Testing FC Layer Forward Pass for Party " << mpc.party << "/" << mpc.M << " ---" << std::endl;

    using T = uint64_t;
    const int F = 16;
    const int K_INT = 15;
    const int IN_BW = 64;
    const int OUT_BW = 48;

    using FixIn = FCLayer<T, IN_BW, OUT_BW, F, K_INT>::FixIn;
    using FixBias = FCLayer<T, IN_BW, OUT_BW, F, K_INT>::FixBias;
    using InputTensor = FCLayer<T, IN_BW, OUT_BW, F, K_INT>::InputTensor;
    using WeightTensor = FCLayer<T, IN_BW, OUT_BW, F, K_INT>::WeightTensor;
    using BiasTensor = FCLayer<T, IN_BW, OUT_BW, F, K_INT>::BiasTensor;

    FCLayerParams params = {5, 2, 3, 4, false, false, 0}; // B, M, N, K, use_bias=false, reconstructed_input, trunc_bwd
    FCLayer<T, IN_BW, OUT_BW, F, K_INT> fc_layer(params);

    std::cout << "FC Layer initialized" << std::endl;
    
    std::cout << "Initializing Weights" << std::endl;

    // 1. Initialize Weights (still pass bias data, but it won't be used)
    uint8_t* weights_buffer = nullptr;
    uint8_t* weights_buffer_ptr = nullptr;
    if (mpc.party == 0) {
        float w_data[] = {1.0, -1.0, 0.5, 1.2, -0.8, 1.2, 1.5, -0.5, 0.5, -0.5, 1.0, -1.0};
        float y_data[] = {0.1, -0.2, 0.3, -0.4};
        weights_buffer = new uint8_t[sizeof(w_data) + sizeof(y_data)];
        memcpy(weights_buffer, w_data, sizeof(w_data));
        memcpy(weights_buffer + sizeof(w_data), y_data, sizeof(y_data));
        weights_buffer_ptr = weights_buffer;
    }
    fc_layer.initWeights(&weights_buffer_ptr, true);
    if (mpc.party == 0) {
        delete[] weights_buffer;
    }
    
    std::cout << "Weights initialized" << std::endl;

    // Loading randomness is now handled by the FCLayer itself, using the MPC object.
    std::cout << "Loading Randomness" << std::endl;
    fc_layer.readForwardRandomness(mpc);
    std::cout << "Randomness loaded" << std::endl;

    // 3. Secret Share Input
    InputTensor x_plain(params.B, params.M, params.N);
    if (mpc.party == 0) {
        x_plain.setZero();
        Eigen::Tensor<FixIn, 2, Eigen::RowMajor> x_slice(params.M, params.N);
        x_slice.setValues({ {FixIn(0.5), FixIn(-1.0), FixIn(1.5)}, {FixIn(-2.0), FixIn(2.5), FixIn(-3.0)} });
        // Manually copy the 2D slice into the first batch of the 3D tensor
        for (int i = 0; i < params.M; ++i) {
            for (int j = 0; j < params.N; ++j) {
                x_plain(0, i, j) = x_slice(i, j);
            }
        }
    } else {
        x_plain.setZero();
    }
    auto x_share = secret_share_tensor(x_plain);

    // 4. Execute Forward Pass
    auto y_share = fc_layer.forward<0>(x_share);
    auto y_reconstructed = reconstruct_tensor(y_share);

    // 5. Plaintext Verification
    if (mpc.party == 0) {
        WeightTensor W_plain(params.N, params.K);
        W_plain.setValues({ {FixIn(1.0), FixIn(-1.0), FixIn(0.5), FixIn(1.2)}, {FixIn(-0.8), FixIn(1.2), FixIn(1.5), FixIn(-0.5)}, {FixIn(0.5), FixIn(-0.5), FixIn(1.0), FixIn(-1.0)} });
        
        auto expected_mul_expr = tensor_mul(x_plain, W_plain);
        
        FixTensor<T, IN_BW, F, K_INT, 3> expected_mul(params.B, params.M, params.K);
        for(int i = 0; i < expected_mul.size(); ++i) {
            expected_mul.data()[i].val = expected_mul_expr.data()[i].val;
        }
        
        auto expected_trunc = truncate_reduce_tensor(expected_mul);
        auto expected_y_final = change_bitwidth<OUT_BW, F, K_INT>(expected_trunc);

        std::cout << "Reconstructed Output:\n" << y_reconstructed << std::endl;
        std::cout << "Expected Output:\n" << expected_y_final << std::endl;

        for (int i = 0; i < y_reconstructed.size(); ++i) {
            assert(std::abs(y_reconstructed.data()[i].to_float<double>() - expected_y_final.data()[i].to_float<double>()) < 1e-3);
        }
    }
    
    std::cout << "Party " << mpc.party << " FC Layer Forward test passed!" << std::endl;
}

void test_fc_backward(MPC& mpc) {
    std::cout << "--- Testing FC Layer Backward Pass for Party " << mpc.party << "/" << mpc.M << " ---" << std::endl;

    using T = uint64_t;
    const int F = 16;
    const int K_INT = 15;
    const int IN_BW = 64;
    const int OUT_BW = 48;

    using FixIn = FCLayer<T, IN_BW, OUT_BW, F, K_INT>::FixIn;
    using WeightTensor = FCLayer<T, IN_BW, OUT_BW, F, K_INT>::WeightTensor;
    using IncomingGradTensor = FCLayer<T, IN_BW, OUT_BW, F, K_INT>::IncomingGradTensor;

    FCLayerParams params = {5, 2, 3, 4, false, false, 0};
    FCLayer<T, IN_BW, OUT_BW, F, K_INT> fc_layer(params);
    
    // 1. Initialize Weights (must be same as forward pass)
    // We need to do this to have the weights ready for the backward pass.
    // The forward pass is not run here, but weights are needed.
    uint8_t* weights_buffer = nullptr;
    uint8_t* weights_buffer_ptr = nullptr;
    if (mpc.party == 0) {
        float w_data[] = {1.0, -1.0, 0.5, 1.2, -0.8, 1.2, 1.5, -0.5, 0.5, -0.5, 1.0, -1.0};
        weights_buffer = new uint8_t[sizeof(w_data)];
        memcpy(weights_buffer, w_data, sizeof(w_data));
        weights_buffer_ptr = weights_buffer;
    }
    fc_layer.initWeights(&weights_buffer_ptr, true);
    if (mpc.party == 0) delete[] weights_buffer;

    // 2. Load backward pass randomness
    // First, skip forward randomness to get to the backward part.
    mpc.random_data_idx += fc_layer.getForwardRandomnessSize();
    fc_layer.readBackwardRandomness(mpc);
    std::cout << "Backward randomness loaded" << std::endl;

    // 3. Secret Share a dummy incoming gradient
    IncomingGradTensor incoming_grad_plain(params.B, params.M, params.K);
    if (mpc.party == 0) {
        incoming_grad_plain.setZero();
        Eigen::Tensor<FixIn, 2, Eigen::RowMajor> og_slice(params.M, params.K);
        og_slice.setValues({{FixIn(0.1), FixIn(0.2), FixIn(0.3), FixIn(0.4)}, {FixIn(-0.1), FixIn(-0.2), FixIn(-0.3), FixIn(-0.4)}});
        for (int i = 0; i < params.M; ++i) {
            for (int j = 0; j < params.K; ++j) {
                incoming_grad_plain(0, i, j) = og_slice(i, j);
            }
        }
    } else {
        incoming_grad_plain.setZero();
    }
    auto incoming_grad_share = secret_share_tensor(incoming_grad_plain);

    // 4. Execute Backward Pass
    auto outgoing_grad_share = fc_layer.backward(incoming_grad_share);
    auto outgoing_grad_reconstructed = reconstruct_tensor(outgoing_grad_share);

    // 5. Plaintext Verification
    if (mpc.party == 0) {
        WeightTensor W_plain(params.N, params.K);
        W_plain.setValues({ {FixIn(1.0), FixIn(-1.0), FixIn(0.5), FixIn(1.2)}, {FixIn(-0.8), FixIn(1.2), FixIn(1.5), FixIn(-0.5)}, {FixIn(0.5), FixIn(-0.5), FixIn(1.0), FixIn(-1.0)} });
        WeightTensor W_plain_T = W_plain.shuffle(Eigen::array<int, 2>{1, 0});
        
        auto expected_outgoing_grad_expr = tensor_mul(incoming_grad_plain, W_plain_T);
        auto expected_outgoing_grad = truncate_tensor(expected_outgoing_grad_expr);

        std::cout << "Reconstructed Outgoing Grad (.val, first batch):\n";
        for(int i=0; i<params.M; ++i) {
            for(int j=0; j<params.N; ++j) std::cout << outgoing_grad_reconstructed(0,i,j).val << " ";
            std::cout << std::endl;
        }
        std::cout << "Expected Outgoing Grad (.val, first batch):\n";
        for(int i=0; i<params.M; ++i) {
            for(int j=0; j<params.N; ++j) std::cout << expected_outgoing_grad(0,i,j).val << " ";
            std::cout << std::endl;
        }

        for (int i = 0; i < params.M; ++i) {
            for (int j = 0; j < params.N; ++j) {
                assert(std::abs(static_cast<int64_t>(outgoing_grad_reconstructed(0, i, j).val) - static_cast<int64_t>(expected_outgoing_grad(0, i, j).val)) <= 1);
            }
        }
    }

    std::cout << "Party " << mpc.party << " FC Layer Backward test passed!" << std::endl;
}

int main(int argc, char** argv) {
    if (argc < 2) {
        std::cout << "Usage: " << argv[0] << " <party>" << std::endl;
        return 1;
    }
    int party = std::atoi(argv[1]);
    MPC mpc(2, party);
    
    std::string randomness_path = "randomness/P" + std::to_string(party) + "/fc_random_data.bin";
    mpc.load_random_data(randomness_path);
    
    std::vector<std::string> addrs = {"127.0.0.1", "127.0.0.1"};
    mpc.connect(addrs, 9001);

    test_fc_forward(mpc);
    test_fc_backward(mpc);

    return 0;
}
