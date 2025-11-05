#include "mpc/fix_tensor.h"
#include <iostream>
#include <cassert>
#include <cmath>

void test_concatenate_manual() {
    std::cout << "--- Testing Manual Concatenate ---" << std::endl;

    using T = uint64_t;
    const int BW = 48;
    const int F = 16;
    const int K = 15;
    using TestFix = Fix<T, BW, F, K>;
    using TestTensor = FixTensor<T, BW, F, K, 2>;

    // 1. Test concatenation along axis 0 (stacking rows)
    std::cout << "Testing axis 0 concatenation..." << std::endl;
    TestTensor a(2, 3);
    a(0, 0) = TestFix(1.0); a(0, 1) = TestFix(2.0); a(0, 2) = TestFix(3.0);
    a(1, 0) = TestFix(4.0); a(1, 1) = TestFix(5.0); a(1, 2) = TestFix(6.0);

    TestTensor b(1, 3);
    b(0, 0) = TestFix(7.0); b(0, 1) = TestFix(8.0); b(0, 2) = TestFix(9.0);

    // Manual Concatenation
    TestTensor result0(a.dimension(0) + b.dimension(0), a.dimension(1));
    for (int i = 0; i < a.dimension(0); ++i) {
        for (int j = 0; j < a.dimension(1); ++j) {
            result0(i, j) = a(i, j);
        }
    }
    for (int i = 0; i < b.dimension(0); ++i) {
        for (int j = 0; j < b.dimension(1); ++j) {
            result0(a.dimension(0) + i, j) = b(i, j);
        }
    }

    assert(result0.dimension(0) == 3);
    assert(result0.dimension(1) == 3);
    assert(std::abs(result0(2, 0).to_float<double>() - 7.0) < 1e-6);
    assert(std::abs(result0(2, 1).to_float<double>() - 8.0) < 1e-6);
    std::cout << "Axis 0 concatenation passed." << std::endl;

    // 2. Test concatenation along axis 1 (stacking columns)
    std::cout << "Testing axis 1 concatenation..." << std::endl;
    TestTensor c(2, 1);
    c(0, 0) = TestFix(10.0);
    c(1, 0) = TestFix(11.0);

    // Manual Concatenation
    TestTensor result1(a.dimension(0), a.dimension(1) + c.dimension(1));
    for (int i = 0; i < a.dimension(0); ++i) {
        for (int j = 0; j < a.dimension(1); ++j) {
            result1(i, j) = a(i, j);
        }
    }
    for (int i = 0; i < c.dimension(0); ++i) {
        for (int j = 0; j < c.dimension(1); ++j) {
            result1(i, a.dimension(1) + j) = c(i, j);
        }
    }
    
    assert(result1.dimension(0) == 2);
    assert(result1.dimension(1) == 4);
    assert(std::abs(result1(0, 3).to_float<double>() - 10.0) < 1e-6);
    assert(std::abs(result1(1, 3).to_float<double>() - 11.0) < 1e-6);
    std::cout << "Axis 1 concatenation passed." << std::endl;
}

template <typename T, int BW, int F, int K_INT>
void test_set_random_bit_length(int rows, int cols) {
    std::cout << "--- Testing FixTensor<" << BW << " bits>(" << rows << "x" << cols << ") ---" << std::endl;

    FixTensor<T, BW, F, K_INT, 2> tensor(rows, cols);
    tensor.setRandom();

    bool all_bits_correct = true;
    // The mask will have 1s for all bits beyond BW.
    // e.g., if BW=48, T=uint64_t, mask will be 0xFFFF000000000000
    T mask = (static_cast<T>(-1) << BW); 

    for (int i = 0; i < tensor.size(); ++i) {
        T raw_value = tensor.data()[i].val;
        // Check if any bits are set in the masked region (i.e., beyond BW)
        if ((raw_value & mask) != 0) {
            all_bits_correct = false;
            std::cout << "Error: Value exceeds bit width " << BW << " at index " << i << std::endl;
            std::cout << "Value (hex): 0x" << std::hex << raw_value << std::dec << std::endl;
            break; 
        }
    }

    if (all_bits_correct) {
        std::cout << "Success: All random numbers are within the specified bit width (" << BW << " bits)." << std::endl;
    } else {
        std::cout << "Failure: At least one random number exceeded the bit width." << std::endl;
    }

    std::cout << "Sample values (hex):" << std::endl;
    int samples_to_show = std::min(5, (int)tensor.size());
    for (int i = 0; i < samples_to_show; ++i) {
        std::cout << "  0x" << std::hex << tensor.data()[i].val << std::dec << std::endl;
    }
    std::cout << std::endl;
}

int main() {
    test_concatenate_manual();
    std::cout << "FixTensor ops test passed!" << std::endl;

    using T = uint64_t;
    const int F = 16;
    const int K_INT = 15;

    // Test with a bit width that is less than the underlying type T
    const int BW_48 = 48;
    test_set_random_bit_length<T, BW_48, F, K_INT>(1, 1);
    test_set_random_bit_length<T, BW_48, F, K_INT>(3, 4);
    test_set_random_bit_length<T, BW_48, F, K_INT>(5, 5);

    // Test with a bit width that is equal to the underlying type T
    // In this case, the mask check is trivial but good to verify
    const int BW_64 = 64;
    test_set_random_bit_length<T, BW_64, F, K_INT>(2, 2);

    return 0;
}
