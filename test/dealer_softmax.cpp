#include <iostream>
#include <fstream>
#include <vector>
#include <cassert>
#include "nn/Softmax.h"
#include "mpc/mpc.h"
#include "mpc/fix_tensor.h"
#include "mpc/tensor_ops.h"

void dealer_generate_softmax_randomness() {
    std::cout << "Softmax Dealer starting..." << std::endl;
    
    system("mkdir -p ./randomness/P0 && mkdir -p ./randomness/P1");

    using T = uint64_t;
    const int BW = 64;
    const int smallBW = 44;
    const int F = 20;
    const int K_INT = 15;

    SoftmaxLayerParams softmax_layer_params;
    softmax_layer_params.B = 5;
    SoftmaxLayer<T, BW, smallBW, F, K_INT> softmax_layer(softmax_layer_params);
    
    size_t randomness_size = softmax_layer.getRandomnessSize();
    uint8_t* p0_data = new uint8_t[randomness_size];
    uint8_t* p1_data = new uint8_t[randomness_size];
    Buffer randomness_buf_0(p0_data);
    Buffer randomness_buf_1(p1_data);
    
    softmax_layer.generate_randomness(randomness_buf_0, randomness_buf_1);

    size_t p0_offset = randomness_buf_0.ptr - p0_data;
    size_t p1_offset = randomness_buf_1.ptr - p1_data;
    if (p0_offset != randomness_size || p1_offset != randomness_size) {
        std::cerr << "Error: Randomness generation did not write the expected number of bytes." << std::endl;
        std::cerr << "Expected: " << randomness_size << ", P0 wrote: " << p0_offset << ", P1 wrote: " << p1_offset << std::endl;
        return;
    }

    std::string p0_randomness_path = "./randomness/P0/softmax_randomness.bin";
    std::string p1_randomness_path = "./randomness/P1/softmax_randomness.bin";
    
    std::ofstream p0_randomness_file(p0_randomness_path, std::ios::binary);
    if (!p0_randomness_file) {
        std::cerr << "Error opening file for party 0." << std::endl;
        return;
    }
    p0_randomness_file.write(reinterpret_cast<const char*>(p0_data), randomness_size);
    p0_randomness_file.close();
    
    std::ofstream p1_randomness_file(p1_randomness_path, std::ios::binary);
    if (!p1_randomness_file) {
        std::cerr << "Error opening file for party 1." << std::endl;
        return;
    }
    p1_randomness_file.write(reinterpret_cast<const char*>(p1_data), randomness_size);
    p1_randomness_file.close();

    std::cout << "Softmax Dealer finished successfully." << std::endl;
}

int main() {
    dealer_generate_softmax_randomness();
    return 0;
}
