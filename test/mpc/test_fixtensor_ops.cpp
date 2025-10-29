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

int main() {
    test_concatenate_manual();
    std::cout << "FixTensor ops test passed!" << std::endl;
    return 0;
}
