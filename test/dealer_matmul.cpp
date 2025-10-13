#include "mpc/mpc.h"
#include <iostream>
#include <fstream>

// This dealer generates Beaver triples for a 2x3 and 3x4 matrix multiplication.
void generate_matmul_randomness() {
    using T = uint64_t;
    const int F = 16;
    const int K_INT = 15;
    const int BW = 64;

    using Fix = Fix<T, BW, F, K_INT>;
    using Tensor2D = FixTensor<T, BW, F, K_INT, 2>;
    
    const int M = 2;
    const int N = 3;
    const int P = 4;

    // Generate U, V, Z
    Tensor2D U(M, N);
    Tensor2D V(N, P);
    U.initialize(); // Random values
    V.initialize(); // Random values
    auto Z_full = tensor_mul(U, V);

    // Manually secret share them (T0 is random, T1 = T - T0)
    Tensor2D U0(M, N); U0.initialize();
    Tensor2D U1 = U - U0;
    Tensor2D V0(N, P); V0.initialize();
    Tensor2D V1 = V - V0;
    Tensor2D Z0(M, P); Z0.initialize();
    Tensor2D Z1 = Z_full - Z0;

    // Write shares to files
    std::ofstream out0("randomness/P0/matmul_random_data.bin", std::ios::binary);
    out0.write(reinterpret_cast<const char*>(U0.data()), U0.size() * sizeof(Fix));
    out0.write(reinterpret_cast<const char*>(V0.data()), V0.size() * sizeof(Fix));
    out0.write(reinterpret_cast<const char*>(Z0.data()), Z0.size() * sizeof(Fix));
    out0.close();

    std::ofstream out1("randomness/P1/matmul_random_data.bin", std::ios::binary);
    out1.write(reinterpret_cast<const char*>(U1.data()), U1.size() * sizeof(Fix));
    out1.write(reinterpret_cast<const char*>(V1.data()), V1.size() * sizeof(Fix));
    out1.write(reinterpret_cast<const char*>(Z1.data()), Z1.size() * sizeof(Fix));
    out1.close();
    
    std::cout << "Matmul randomness generated successfully." << std::endl;
}

int main() {
    generate_matmul_randomness();
    return 0;
}
