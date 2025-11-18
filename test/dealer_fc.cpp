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

    // Parameters for the FC layer (B, in_dim, out_dim, use_bias, reconstructed_input, trunc_bwd)
    FCLayerParams params_fc = {5, 2, 3, false, false};
    FCLayer<T_fc, IN_BW_fc, OUT_BW_fc, F_fc, K_INT_fc> fc_layer(params_fc);

    // --- Generate Forward Randomness ---
    size_t randomness_size = fc_layer.getRandomnessSize();
    std::cout << "Randomness size: " << randomness_size << " bytes" << std::endl;
    uint8_t* p0_data = new uint8_t[randomness_size];
    uint8_t* p1_data = new uint8_t[randomness_size];
    Buffer randomness_buf_0(p0_data);
    Buffer randomness_buf_1(p1_data);
    fc_layer.generate_randomness(randomness_buf_0, randomness_buf_1);

    std::string p0_randomness_path = "./randomness/P0/fc_randomness.bin";
    std::string p1_randomness_path = "./randomness/P1/fc_randomness.bin";
    std::ofstream p0_randomness_file(p0_randomness_path, std::ios::binary);
    if (!p0_randomness_file) {
        std::cerr << "Error opening file for party 0." << std::endl;
        return; // Changed from return 1 to return
    }
    p0_randomness_file.write(reinterpret_cast<const char*>(p0_data), randomness_size);
    p0_randomness_file.close();
    std::ofstream p1_randomness_file(p1_randomness_path, std::ios::binary);
    if (!p1_randomness_file) {
        std::cerr << "Error opening file for party 1." << std::endl;
        return; // Changed from return 1 to return
    }
    p1_randomness_file.write(reinterpret_cast<const char*>(p1_data), randomness_size);
    p1_randomness_file.close();

    // 6. Clean up allocated memory
    delete[] p0_data;
    delete[] p1_data;

    std::cout << "FC Dealer finished successfully." << std::endl;
}

int main() {
    dealer_generate_fc_randomness();
    return 0;
}
