#include "nn/FC.h"
#include <iostream>
#include <vector>
#include <cassert>

void test_fc_forward(MPC& mpc) {
    std::cout << "--- Testing FC Layer Forward Pass for Party " << mpc.party << "/" << mpc.M << " ---" << std::endl;

    using T = int64_t;
    const int F = 16;
    const int K_INT = 15;
    const int IN_BW = 64;
    const int OUT_BW = 48;

    using FixIn = FCLayer<T, IN_BW, OUT_BW, F, K_INT>::FixIn;
    using FixBias = FCLayer<T, IN_BW, OUT_BW, F, K_INT>::FixBias;
    using InputTensor = FCLayer<T, IN_BW, OUT_BW, F, K_INT>::InputTensor;
    using WeightTensor = FCLayer<T, IN_BW, OUT_BW, F, K_INT>::WeightTensor;
    using BiasTensor = FCLayer<T, IN_BW, OUT_BW, F, K_INT>::BiasTensor;

    FCLayerParams params = {1, 2, 3, 4, false, false, 0}; // B, M, N, K, use_bias=false, reconstructed_input, trunc_bwd
    FCLayer<T, IN_BW, OUT_BW, F, K_INT> fc_layer(params);

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
    
    // 2. Load Randomness from MPC buffer and pass to layer
    uint8_t* random_data_ptr = mpc.random_data.data();
    fc_layer.readForwardRandomness(random_data_ptr);

    // 3. Secret Share Input
    InputTensor x_plain(params.M, params.N);
    if (mpc.party == 0) {
        x_plain.setValues({ {FixIn(0.5), FixIn(-1.0), FixIn(1.5)}, {FixIn(-2.0), FixIn(2.5), FixIn(-3.0)} });
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
        
        FixTensor<T, IN_BW, 2*F, K_INT, 2> expected_mul(params.M, params.K);
        for(int i = 0; i < expected_mul.size(); ++i) {
            expected_mul.data()[i].val = expected_mul_expr.data()[i].val;
        }
        
        auto expected_trunc = truncate_reduce_tensor_plain<F>(expected_mul);
        auto expected_y_final = change_bitwidth_plain<OUT_BW, F, K_INT>(expected_trunc);

        std::cout << "Reconstructed Output:\n" << y_reconstructed << std::endl;
        std::cout << "Expected Output:\n" << expected_y_final << std::endl;

        for (int i = 0; i < y_reconstructed.size(); ++i) {
            assert(std::abs(y_reconstructed.data()[i].to_float<double>() - expected_y_final.data()[i].to_float<double>()) < 1e-3);
        }
    }
    
    std::cout << "Party " << mpc.party << " FC Layer Forward test passed!" << std::endl;
}

int main(int argc, char** argv) {
    if (argc < 2) {
        std::cout << "Usage: " << argv[0] << " <party>" << std::endl;
        return 1;
    }
    int party = std::atoi(argv[1]);
    MPC mpc(2, party);
    
    std::vector<std::string> addrs = {"127.0.0.1", "127.0.0.1"};
    mpc.connect(addrs, 9001);

    std::string filename = (party == 0) ? "./randomness/P0/random_data.bin" : "./randomness/P1/random_data.bin";
    mpc.load_random_data(filename);

    test_fc_forward(mpc);

    return 0;
}
