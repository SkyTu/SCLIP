#include "nn/L2NormParallel.h"
#include "mpc/mpc.h"
#include "mpc/tensor_ops.h"
#include <iostream>
#include <vector>
#include <cassert>
#include <fstream>
#include <cmath>

void test_l2norm_parallel(MPC& mpc) {
    std::cout << "--- Testing L2NormParallel Layer Forward Pass for Party " << mpc.party << "/" << mpc.M << " ---" << std::endl;

    using T = uint64_t;
    const int IN_BW = 48;
    const int OUT_BW = 64;
    const int F = 16;
    const int K_INT = 15;

    using FixIn = Fix<T, IN_BW, F, K_INT>;
    using InputTensor = L2NormLayer<T, IN_BW, OUT_BW, F, K_INT>::InputTensor;
    
    L2NormLayerParams params = {5, 10}; // B=5, in_dim=10
    L2NormLayer<T, IN_BW, OUT_BW, F, K_INT> l2_layer(params);

    std::cout << "L2NormParallel Layer initialized" << std::endl;

    // 1. Secret Share Inputs
    InputTensor image_plain(params.B, params.in_dim);
    InputTensor text_plain(params.B, params.in_dim);
    InputTensor image_share(params.B, params.in_dim);
    InputTensor text_share(params.B, params.in_dim);

    if (mpc.party == 0) {
        // Use Eigen's random capabilities to generate test data
        for(int i = 0; i < params.B; ++i) {
            for(int j = 0; j < params.in_dim; ++j) {
                image_plain(i, j) = FixIn(double(i+j)/10.0);
                text_plain(i, j) = FixIn(double(i-j)/10.0);
            }
        }
        // Party 0 holds the full plaintext as its share
        image_share = image_plain;
        text_share = text_plain;
    } else {
        // Other parties have zero shares
        image_share.setZero();
        text_share.setZero();
    }
    
    std::cout << "Loading Randomness" << std::endl;
    l2_layer.read_randomness(mpc);
    std::cout << "Randomness loaded" << std::endl;

    std::cout << "Executing forward pass" << std::endl;
    auto y_share = l2_layer.forward(image_share, text_share);
    std::cout << "Forward pass executed" << std::endl;

    auto y_reconstructed = reconstruct_tensor(y_share);

    // 6. Plaintext Verification
    if (mpc.party == 0) {
        // Re-shape plaintext inputs for stacking, just like in the layer
        FixTensor<T, IN_BW, F, K_INT, 3> all_plain_3d(2, params.B, params.in_dim);
        for(int i = 0; i < params.B; ++i) {
            for(int j = 0; j < params.in_dim; ++j) {
                all_plain_3d(0, i, j) = image_plain(i, j);
                all_plain_3d(1, i, j) = text_plain(i, j);
            }
        }

        // Plaintext L2 Normalization
        FixTensor<T, OUT_BW, F, K_INT, 3> expected_y(2, params.B, params.in_dim);
        for (int i = 0; i < 2; ++i) {
            for (int b = 0; b < params.B; ++b) {
                double norm_sq = 0.0;
                for (int d = 0; d < params.in_dim; ++d) {
                    norm_sq += all_plain_3d(i, b, d).to_float<double>() * all_plain_3d(i, b, d).to_float<double>();
                }
                double norm = std::sqrt(norm_sq) + 1e-12; // Add epsilon for stability
                for (int d = 0; d < params.in_dim; ++d) {
                    expected_y(i, b, d) = Fix<T, OUT_BW, F, K_INT>(all_plain_3d(i, b, d).to_float<double>() / norm);
                }
            }
        }
        
        std::cout << "Reconstructed Output:\n" << y_reconstructed << std::endl;
        std::cout << "Expected Output:\n" << expected_y << std::endl;

        for (int i = 0; i < y_reconstructed.size(); ++i) {
            assert(std::abs(y_reconstructed.data()[i].to_float<double>() - expected_y.data()[i].to_float<double>()) < 1e-3);
        }
        std::cout << "L2NormParallel forward verification passed." << std::endl;
    }
    
    std::cout << "Party " << mpc.party << " L2NormParallel Layer test passed!" << std::endl;
}

bool file_exists(const std::string& path) {
    std::ifstream file(path);
    return file.good();
}

int main(int argc, char** argv) {
    if (argc < 2) {
        std::cout << "Usage: " << argv[0] << " <party>" << std::endl;
        return 1;
    }
    int party = std::atoi(argv[1]);
    MPC mpc(2, party);
    
    std::string randomness_path = "./randomness/P" + std::to_string(party) + "/l2norm_parallel_randomness.bin";
    if (!file_exists(randomness_path)) {
        std::cerr << "Randomness file not found: " << randomness_path << std::endl;
        return 1;
    }
    mpc.load_random_data(randomness_path);
    
    std::vector<std::string> addrs = {"127.0.0.1", "127.0.0.1"};
    mpc.connect(addrs, 9001);

    test_l2norm_parallel(mpc);

    return 0;
}
