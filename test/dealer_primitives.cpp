#include <iostream>
#include <fstream>
#include <vector>
#include "mpc/fix_tensor.h"
#include "utils/random.h"
#include "mpc/tensor_ops.h" // For tensor_mul

using T = uint64_t;
const int BW = 64;
const int F = 16;
const int K = 31;

// Helper to generate Beaver triples for matrix multiplication
template<typename TensorA, typename TensorB, typename TensorC>
void generate_matmul_triples(std::vector<uint8_t>& p0_buffer, std::vector<uint8_t>& p1_buffer) {
    TensorA U; U.initialize();
    TensorB V; V.initialize();
    TensorC Z = tensor_mul(U, V);

    auto [U0, U1] = secret_share_into_two(U);
    auto [V0, V1] = secret_share_into_two(V);
    auto [Z0, Z1] = secret_share_into_two(Z);
    
    // Append to buffers...
}

// Main function
int main() {
    // MatMul 2D
    using MatMulTensorA = FixTensor<T, BW, F, K, 2>;
    using MatMulTensorB = FixTensor<T, BW, F, K, 2>;
    using MatMulTensorC = FixTensor<T, BW, 2*F, K, 2>; // Note: 2*F
    // ... generate ...

    // MatMul 3D
    using MatMul3DTensorA = FixTensor<T, BW, F, K, 3>;
    using MatMul3DTensorC = FixTensor<T, BW, 2*F, K, 3>;
    // ... generate ...

    // ... All other randomness generations for other tests ...
    
    // Write p0_buffer and p1_buffer to files
}
