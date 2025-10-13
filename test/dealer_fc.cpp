#include <iostream>
#include <fstream>
#include <vector>
#include <cassert>
#include "nn/FC.h"
#include "mpc/fix_tensor.h"

int main() {
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

    // 1. Calculate the size of randomness needed for the FC layer
    size_t fwd_randomness_size = fc_layer.getForwardRandomnessSize();
    size_t bwd_randomness_size = fc_layer.getBackwardRandomnessSize();
    size_t total_randomness_size = fwd_randomness_size + bwd_randomness_size;
    std::cout << "Total FC randomness size per party: " << total_randomness_size << " bytes." << std::endl;

    // 2. Allocate buffers for each party's share of the randomness
    uint8_t* p0_data = new uint8_t[total_randomness_size];
    uint8_t* p1_data = new uint8_t[total_randomness_size];
    uint8_t* p0_ptr = p0_data;
    uint8_t* p1_ptr = p1_data;

    // 3. Generate the randomness and write the shares to the buffers
    fc_layer.dealer_generate_randomness(p0_ptr, p1_ptr);
    fc_layer.dealer_generate_backward_randomness(p0_ptr, p1_ptr);

    // 4. Assert that we wrote the exact calculated size
    assert(static_cast<size_t>(p0_ptr - p0_data) == total_randomness_size);
    assert(static_cast<size_t>(p1_ptr - p1_data) == total_randomness_size);

    // 5. Write the buffers to separate files for each party
    std::ofstream p0_file("./randomness/P0/fc_random_data.bin", std::ios::binary);
    if (!p0_file) {
        std::cerr << "Error opening file for party 0." << std::endl;
        return 1;
    }
    p0_file.write(reinterpret_cast<const char*>(p0_data), total_randomness_size);
    p0_file.close();

    std::ofstream p1_file("./randomness/P1/fc_random_data.bin", std::ios::binary);
    if (!p1_file) {
        std::cerr << "Error opening file for party 1." << std::endl;
        return 1;
    }
    p1_file.write(reinterpret_cast<const char*>(p1_data), total_randomness_size);
    p1_file.close();

    // 6. Clean up allocated memory
    delete[] p0_data;
    delete[] p1_data;

    std::cout << "FC Dealer finished successfully." << std::endl;
    return 0;
}
