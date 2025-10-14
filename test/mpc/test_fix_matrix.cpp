#include "mpc/fix_tensor.h"
#include "mpc/truncate.h"
#include "mpc/tensor_ops.h"
#include <iostream>
#include <cassert>
#include <vector>
#include <cmath> // Required for floor

const int BW = 64;
const int F = 16;
const int K = 31; // 1 + 16 + 31 + 16 = 64

using MyFix = Fix<uint64_t, BW, F, K>;
using MyFixTensor2 = FixTensor<uint64_t, BW, F, K, 2>;
using MyFixTensor3 = FixTensor<uint64_t, BW, F, K, 3>;


void test_tensor_operations() {
    std::cout << "\n--- Testing Tensor Operations with Random Numbers ---" << std::endl;

    // --- Test Basic Arithmetic (2D) ---
    MyFixTensor2 A(2, 2);
    A.initialize(10);
    A.print("Random 2D Tensor A:");

    MyFixTensor2 B(2, 2);
    B.initialize(10);
    B.print("Random 2D Tensor B:");

    std::cout << "\nTesting Addition..." << std::endl;
    MyFixTensor2 C = A + B;
    C.print("A + B =");

    for (int i = 0; i < 2; ++i) {
        for (int j = 0; j < 2; ++j) {
            double expected = A(i, j).to_float<double>() + B(i, j).to_float<double>();
            double actual = C(i, j).to_float<double>();
            if (std::abs(actual - expected) >= 1) {
                std::cout << "Addition failed at (" << i << ", " << j << "):" << std::endl;
                std::cout << "  A(" << i << ", " << j << ") = " << A(i, j).to_float<double>() << " (raw: " << A(i, j).val << ")" << std::endl;
                std::cout << "  B(" << i << ", " << j << ") = " << B(i, j).to_float<double>() << " (raw: " << B(i, j).val << ")" << std::endl;
                std::cout << "  Expected: " << expected << std::endl;
                std::cout << "  Actual:   " << actual << " (raw: " << C(i, j).val << ")" << std::endl;
            }
            assert(std::abs(actual - expected) < 1);
        }
    }
    std::cout << "Addition test passed." << std::endl;

    // --- Test 2D x 2D Multiplication ---
    std::cout << "\nTesting 2D x 2D Multiplication..." << std::endl;
    MyFixTensor2 E_intermediate = tensor_mul(A, B);
    MyFixTensor2 E = E_intermediate.unaryExpr([](const MyFix& x){ return x.trunc(F); });
    E.print("A * B =");
    for (int i = 0; i < 2; ++i) {
        for (int j = 0; j < 2; ++j) {
            double expected = 0;
            for (int l = 0; l < 2; ++l) {
                expected += A(i, l).to_float<double>() * B(l, j).to_float<double>();
            }
            double actual = E(i, j).to_float<double>();
            if (std::abs(actual - expected) >= 1e-1) {
                std::cout << "2D x 2D Multiplication failed at (" << i << ", " << j << ")" << std::endl;
                std::cout << "  Expected: " << expected << std::endl;
                std::cout << "  Actual:   " << actual << std::endl;
            }
            assert(std::abs(actual - expected) < 1e-1);
        }
    }
    std::cout << "2D x 2D Multiplication test passed." << std::endl;

    // --- Test Element-wise Multiplication ---
    std::cout << "\n--- 开始测试 elementwise_mul ---" << std::endl;
    std::cout << "\nTesting Element-wise Multiplication..." << std::endl;
    MyFixTensor2 H_intermediate = A * B;
    MyFixTensor2 H = H_intermediate.unaryExpr([](const MyFix& x){ return x.trunc(F); });
    H.print("A .* B (element-wise) =");
    for (int i = 0; i < 2; ++i) {
        for (int j = 0; j < 2; ++j) {
            double expected = A(i, j).to_float<double>() * B(i, j).to_float<double>();
            double actual = H(i, j).to_float<double>();
            if (std::abs(actual - expected) >= 1e-1) {
                std::cout << "Element-wise Multiplication failed at (" << i << ", " << j << "):" << std::endl;
                std::cout << "  A(" << i << ", " << j << ") = " << A(i, j).to_float<double>() << " (raw: " << A(i, j).val << ")" << std::endl;
                std::cout << "  B(" << i << ", " << j << ") = " << B(i, j).to_float<double>() << " (raw: " << B(i, j).val << ")" << std::endl;
                std::cout << "  Expected: " << expected << std::endl;
                std::cout << "  Actual:   " << actual << " (raw: " << H(i, j).val << ")" << std::endl;
            }
            assert(std::abs(actual - expected) < 1e-1);
        }
    }
    std::cout << "Element-wise Multiplication test passed." << std::endl;
    std::cout << "--- elementwise_mul 测试结束 ---" << std::endl;


    // --- Test 3D x 2D Multiplication ---
    std::cout << "\nTesting 3D x 2D Multiplication..." << std::endl;
    const int batch = 4, m = 2, n = 3, q = 2;
    MyFixTensor3 batch_A(batch, m, n);
    batch_A.initialize(10);
    batch_A.print("Random 3D Tensor batch_A (4x2x3):");

    MyFixTensor2 M(n, q);
    M.initialize(10);
    M.print("Random 2D Tensor M (3x2):");

    MyFixTensor3 batch_C_intermediate = tensor_mul(batch_A, M);
    MyFixTensor3 batch_C = batch_C_intermediate.unaryExpr([](const MyFix& x){ return x.trunc(F); });
    batch_C.print("Result C = batch_A * M:");

    for (int i = 0; i < batch; ++i) {
        for (int j = 0; j < m; ++j) {
            for (int k_val = 0; k_val < q; ++k_val) {
                double expected_val = 0;
                for (int l = 0; l < n; ++l) {
                    expected_val += batch_A(i, j, l).to_float<double>() * M(l, k_val).to_float<double>();
                }
                double actual_val = batch_C(i, j, k_val).to_float<double>();
                if (std::abs(actual_val - expected_val) >= 1e-1) {
                    std::cout << "3D x 2D Multiplication failed at (" << i << ", " << j << ", " << k_val << ")" << std::endl;
                    std::cout << "  Expected: " << expected_val << std::endl;
                    std::cout << "  Actual:   " << actual_val << std::endl;
                }
                assert(std::abs(actual_val - expected_val) < 1e-1);
            }
        }
    }
    std::cout << "3D x 2D Multiplication test passed." << std::endl;

    // --- Test Bit-width Change ---
    std::cout << "\nTesting Bit-width Change..." << std::endl;
    const int NEW_F = 10;
    const int NEW_K = 20; // 1 + 10 + 20 + 10 = 41
    const int NEW_BW = 41;
    using NewFixTensor2 = FixTensor<uint64_t, NEW_BW, NEW_F, NEW_K, 2>;

    A.print("Original Tensor A:");
    NewFixTensor2 A_new_bw = change_bitwidth<NEW_BW, NEW_F, NEW_K>(A);
    A_new_bw.print("A with new bit-width (F=10):");

    for (int i = 0; i < 2; ++i) {
        for (int j = 0; j < 2; ++j) {
            double original_val = A(i, j).to_float<double>();
            double converted_val = A_new_bw(i, j).to_float<double>();
            assert(std::abs(original_val - converted_val) < 1.0 / (1ULL << NEW_F));
        }
    }
    std::cout << "Bit-width change test passed." << std::endl;
}

int main() {
    try {
        test_tensor_operations();
    } catch (const std::exception& e) {
        std::cerr << "An exception occurred: " << e.what() << std::endl;
        return 1;
    }
    return 0;
}
