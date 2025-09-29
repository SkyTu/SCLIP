#include "mpc/fix_tensor.h"
#include <iostream>
#include <cmath> // Required for floor

const int BW = 64;
const int F = 16;
const int K = 31; // 1 + 16 + 31 + 16 = 64

using MyFix3DTensor = FixTensor<uint64_t, BW, F, K, 3>;

void test_tensor_initialization() {
    std::cout << "\n--- Testing Tensor Initialization ---" << std::endl;

    // Test zeros()
    MyFix3DTensor A(2, 3, 4);
    A.zeros();
    for (int i = 0; i < A.size(); ++i) {
        assert(A.data()[i].val == 0);
    }
    std::cout << "zeros() test passed." << std::endl;

    // Test ones()
    A.ones();
    for (int i = 0; i < A.size(); ++i) {
        // 1.0 is represented as 1 << F
        assert(A.data()[i].val == (1ULL << F));
    }
    std::cout << "ones() test passed." << std::endl;

    // Test initialize()
    const int k_int_test = 5;
    A.initialize(k_int_test);
    
    // Validation based on new requirement
    for (int i = 0; i < A.size(); ++i) {
        double float_val = A.data()[i].to_float<double>();
        double int_part = floor(std::abs(float_val));
        assert(int_part < (1ULL << k_int_test));
    }
    std::cout << "initialize() random value range test passed." << std::endl;

    int gap = F + K - k_int_test;
    uint64_t fixed_mask = (1ULL << (BW - 1 - gap)) - 1;
    uint64_t sign_bit_mask = (1ULL << (BW - 1));
    uint64_t allowed_bits_mask = fixed_mask | sign_bit_mask;

     for (int i = 0; i < A.size(); ++i) {
        uint64_t val = A.data()[i].val;
        bool is_negative = (val & sign_bit_mask) != 0;
        
        uint64_t gap_bits = val & (~allowed_bits_mask);

        if (is_negative) {
            // For negative numbers, all gap bits should be 1.
            uint64_t expected_gap_bits = ~allowed_bits_mask;
            assert(gap_bits == expected_gap_bits);
        } else {
            // For positive numbers, all gap bits should be 0.
            assert(gap_bits == 0);
        }
     }
     std::cout << "initialize() gap bitmask test passed." << std::endl;
 
     // Test print()
    std::cout << "\n--- Testing print() ---" << std::endl;
    MyFix3DTensor B(2, 2, 3);
    B.initialize(3, F+K-3);
    B.print("A random 2x2x3 tensor:");
}

int main() {
    try {
        test_tensor_initialization();
    } catch (const std::exception& e) {
        std::cerr << "An exception occurred: " << e.what() << std::endl;
        return 1;
    }
    return 0;
}
