#include <iostream>
#include <fstream>
#include <vector>
#include <cassert>
#include "nn/FC.h"
#include "mpc/mpc.h"
#include "mpc/fix_tensor.h"
#include "mpc/tensor_ops.h"

void dealer_generate_fc_randomness() {
    std::cout << "FC Dealer starting..." << std::endl;
    
    // Create directories for randomness if they don't exist
    system("mkdir -p ./randomness/P0 && mkdir -p ./randomness/P1");

    // --- FC Layer Setup ---
    using T_fc = uint64_t;
    const int F_fc = 16;
    const int K_INT_fc = 15;
    const int IN_BW_fc = 64;
    const int OUT_BW_fc = 48;

    // Parameters for the FC layer (B, M, N, K, use_bias, reconstructed_input, trunc_bwd)
    FCLayerParams params_fc = {5, 2, 3, 4, false, false, 0};
    FCLayer<T_fc, IN_BW_fc, OUT_BW_fc, F_fc, K_INT_fc> fc_layer(params_fc);

    // --- Generate Forward Randomness ---
    size_t fwd_size = fc_layer.getForwardRandomnessSize();
    std::cout << "Forward randomness size: " << fwd_size << " bytes" << std::endl;
    uint8_t* p0_fwd_buf = new uint8_t[fwd_size];
    uint8_t* p1_fwd_buf = new uint8_t[fwd_size];
    uint8_t* p0_fwd_ptr = p0_fwd_buf;
    uint8_t* p1_fwd_ptr = p1_fwd_buf;
    FCLayer<T_fc, IN_BW_fc, OUT_BW_fc, F_fc, K_INT_fc>::InputTensor U_fwd(params_fc.B, params_fc.M, params_fc.N); U_fwd.initialize(K_INT_fc);
    FCLayer<T_fc, IN_BW_fc, OUT_BW_fc, F_fc, K_INT_fc>::WeightTensor V_fwd(params_fc.N, params_fc.K); V_fwd.initialize(K_INT_fc);
    fc_layer.dealer_generate_forward_randomness(p0_fwd_ptr, p1_fwd_ptr, U_fwd, V_fwd);

    // --- Generate Backward Randomness (includes SGD randomness) ---
    size_t bwd_size = fc_layer.getBackwardRandomnessSize();
    std::cout << "Backward randomness size: " << bwd_size << " bytes" << std::endl;
    uint8_t* p0_bwd_buf = new uint8_t[bwd_size];
    uint8_t* p1_bwd_buf = new uint8_t[bwd_size];
    uint8_t* p0_bwd_ptr = p0_bwd_buf;
    uint8_t* p1_bwd_ptr = p1_bwd_buf;
    fc_layer.dealer_generate_backward_randomness(p0_bwd_ptr, p1_bwd_ptr);

    // --- Write to files ---
    std::string p0_fwd_path = "./randomness/P0/fc_fwd_random.bin";
    std::string p1_fwd_path = "./randomness/P1/fc_fwd_random.bin";
    std::ofstream p0_fwd_file(p0_fwd_path, std::ios::binary);
    if (!p0_fwd_file) {
        std::cerr << "Error opening file for party 0." << std::endl;
        return; // Changed from return 1 to return
    }
    p0_fwd_file.write(reinterpret_cast<const char*>(p0_fwd_buf), fwd_size);
    p0_fwd_file.close();

    std::ofstream p1_fwd_file(p1_fwd_path, std::ios::binary);
    if (!p1_fwd_file) {
        std::cerr << "Error opening file for party 1." << std::endl;
        return; // Changed from return 1 to return
    }
    p1_fwd_file.write(reinterpret_cast<const char*>(p1_fwd_buf), fwd_size);
    p1_fwd_file.close();

    std::string p0_bwd_path = "./randomness/P0/fc_bwd_random.bin";
    std::string p1_bwd_path = "./randomness/P1/fc_bwd_random.bin";
    std::ofstream p0_bwd_file(p0_bwd_path, std::ios::binary);
    if (!p0_bwd_file) {
        std::cerr << "Error opening file for party 0." << std::endl;
        return; // Changed from return 1 to return
    }
    p0_bwd_file.write(reinterpret_cast<const char*>(p0_bwd_buf), bwd_size);
    p0_bwd_file.close();

    std::ofstream p1_bwd_file(p1_bwd_path, std::ios::binary);
    if (!p1_bwd_file) {
        std::cerr << "Error opening file for party 1." << std::endl;
        return; // Changed from return 1 to return
    }
    p1_bwd_file.write(reinterpret_cast<const char*>(p1_bwd_buf), bwd_size);
    p1_bwd_file.close();

    // 6. Clean up allocated memory
    delete[] p0_fwd_buf;
    delete[] p1_fwd_buf;
    delete[] p0_bwd_buf;
    delete[] p1_bwd_buf;

    std::cout << "FC Dealer finished successfully." << std::endl;
}

int main() {
    dealer_generate_fc_randomness();
    return 0;
}
