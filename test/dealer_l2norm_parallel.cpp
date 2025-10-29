#include <iostream>
#include <fstream>
#include <vector>
#include "nn/L2NormParallel.h"
#include "mpc/fix_tensor.h"

int main() {
    std::cout << "Dealer for L2NormParallelLayer starting..." << std::endl;
    system("mkdir -p ./randomness/P0 && mkdir -p ./randomness/P1");

    using T = uint64_t;
    const int IN_BW = 48;
    const int OUT_BW = 64;
    const int F = 16;
    const int K_INT = 15;

    L2NormLayerParams params = {5, 10}; // B=5, in_dim=10
    L2NormLayer<T, IN_BW, OUT_BW, F, K_INT> l2_layer(params);

    size_t total_size = l2_layer.get_randomness_size();
    std::cout << "Total randomness size per party: " << total_size << " bytes." << std::endl;

    uint8_t* p0_data = new uint8_t[total_size];
    uint8_t* p1_data = new uint8_t[total_size];

    Buffer p0_buf(p0_data);
    Buffer p1_buf(p1_data);

    std::cout << "Generating randomness" << std::endl;
    l2_layer.generate_randomness(p0_buf, p1_buf);

    // Assert that we wrote the exact calculated size
    std::cout << "Checking randomness size" << std::endl;
    size_t p0_offset = p0_buf.ptr - p0_data;
    size_t p1_offset = p1_buf.ptr - p1_data;
    if (p0_offset != total_size || p1_offset != total_size) {
        std::cerr << "Error: Randomness generation did not write the expected number of bytes." << std::endl;
        std::cerr << "Expected: " << total_size << ", P0 wrote: " << p0_offset << ", P1 wrote: " << p1_offset << std::endl;
        return 1;
    }

    std::ofstream p0_file("./randomness/P0/l2norm_parallel_randomness.bin", std::ios::binary);
    p0_file.write(reinterpret_cast<const char*>(p0_data), total_size);
    p0_file.close();

    std::ofstream p1_file("./randomness/P1/l2norm_parallel_randomness.bin", std::ios::binary);
    p1_file.write(reinterpret_cast<const char*>(p1_data), total_size);
    p1_file.close();

    delete[] p0_data;
    delete[] p1_data;

    std::cout << "Dealer finished successfully." << std::endl;
    return 0;
}
