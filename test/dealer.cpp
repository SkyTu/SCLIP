#include <iostream>
#include <fstream>
#include <vector>
#include <cassert>
#include "mpc/fix_tensor.h"
#include "utils/random.h"
#include "mpc/tensor_ops.h"
#include "mpc/matmul.h"
#include "nn/FC.h"




int main() {
    std::cout << "Dealer starting..." << std::endl;
    system("mkdir -p ./randomness/P0 && mkdir -p ./randomness/P1");

    using T = uint64_t;
    const int BW = 64;
    const int F = 16;
    const int K = 31;
    constexpr int M_BITS = BW - F;
    Random rg;

    // --- FC Layer Setup ---
    using T_fc = uint64_t;
    const int F_fc = 16;
    const int K_INT_fc = 15;
    const int IN_BW_fc = 64;
    const int OUT_BW_fc = 48;
    FCLayerParams params_fc = {5, 2, 3, 4, false, false, 0};
    FCLayer<T_fc, IN_BW_fc, OUT_BW_fc, F_fc, K_INT_fc> fc_layer(params_fc);
    size_t fc_randomness_size = fc_layer.getForwardRandomnessSize();


    // 1. Pre-calculate total size
    size_t total_size = 0;
    const size_t scalar_size = sizeof(T);
    // For test_secure_matmul (2D)
    total_size += (2 * 3) * scalar_size; // U
    total_size += (3 * 2) * scalar_size; // V
    total_size += (2 * 2) * scalar_size; // Z
    // For test_secure_matmul_3d
    total_size += (2 * 2 * 3) * scalar_size; // U
    total_size += (3 * 2) * scalar_size; // V
    total_size += (2 * 2 * 2) * scalar_size; // Z
    // For test_truncate_zero_extend_scalar
    total_size += 3 * scalar_size;
    // For test_truncate_zero_extend_tensor_2d
    total_size += 3 * (20 * 20) * scalar_size;
    // For test_truncate_zero_extend_tensor_3d
    total_size += 3 * (2 * 2 * 2) * scalar_size;
    // For test_elementwise_mul_opt
    total_size += 2 * (3 * 4) * sizeof(uint64_t); // for r_x_m, r_y_m (M_BITS)
    total_size += 8 * (3 * 4) * sizeof(uint64_t); // for r_x_n, r_y_n, and MSBs and products (BW)
    // total_size += fc_randomness_size;
    
    std::cout << "Total size per party: " << total_size << " bytes." << std::endl;

    // 2. Allocate raw uint8_t* buffers
    uint8_t* p0_data = new uint8_t[total_size];
    uint8_t* p1_data = new uint8_t[total_size];

    // 3. Generate randomness and write to buffers
    uint8_t* p0_ptr = p0_data;
    uint8_t* p1_ptr = p1_data;

    // For test_secure_matmul (2D)
    {
        FixTensor<T, BW, F, K, 2> U(2, 3);
        FixTensor<T, BW, F, K, 2> V(3, 2);
        FixTensor<T, BW, F, K, 2> Z(2, 2);
        generate_matmul_randomness<T, BW, F, K, 2, 2, 2>(U, V, Z);
        secret_share_and_write_tensor(U, p0_ptr, p1_ptr);
        secret_share_and_write_tensor(V, p0_ptr, p1_ptr);
        secret_share_and_write_tensor(Z, p0_ptr, p1_ptr);
    }
    
    // For test_secure_matmul_3d
    {
        FixTensor<T, BW, F, K, 3> U(2, 2, 3); U.initialize();
        FixTensor<T, BW, F, K, 2> V(3, 2); V.initialize();
        FixTensor<T, BW, F, K, 3> Z = tensor_mul(U, V);
        secret_share_and_write_tensor(U, p0_ptr, p1_ptr);
        secret_share_and_write_tensor(V, p0_ptr, p1_ptr);
        secret_share_and_write_tensor(Z, p0_ptr, p1_ptr);
    }

    // For test_truncate_zero_extend_scalar
    {
        T r_m_val = rg.template randomGE<T>(1, M_BITS)[0];
        Fix<T, M_BITS, F, K> r_m(r_m_val);
        Fix<T, BW, F, K> r_e(r_m_val); // r_e is the zero-extension of r_m
        Fix<T, BW, F, K> r_msb = r_m.template get_msb<BW, F, K>(); // r_msb is the MSB of r_m
        secret_share_and_write_scalar(r_m, p0_ptr, p1_ptr);
        secret_share_and_write_scalar(r_e, p0_ptr, p1_ptr);
        secret_share_and_write_scalar(r_msb, p0_ptr, p1_ptr);
    }

    // For test_truncate_zero_extend_tensor_2d
    {
        FixTensor<T, M_BITS, F, K, 2> r_m(20, 20); 
        FixTensor<T, BW, F, K, 2> r_e(20, 20);
        FixTensor<T, BW, F, K, 2> r_msb(20, 20);
        for (int i = 0; i < r_m.size(); ++i) {
            T val = rg.template randomGE<T>(1, M_BITS)[0];
            // std::cout << "r_m_val: " << val << std::endl;
            r_m.data()[i] = Fix<T, M_BITS, F, K>(val);
            r_e.data()[i] = r_m.data()[i];
            // std::cout << "r_e_val: " << r_e.data()[i].val << std::endl;
            r_msb.data()[i] = r_m.data()[i].template get_msb<BW, F, K>();
            // std::cout << "r_msb_val: " << r_msb.data()[i].val << std::endl;
        }
        secret_share_and_write_tensor(r_m, p0_ptr, p1_ptr);
        secret_share_and_write_tensor(r_e, p0_ptr, p1_ptr);
        secret_share_and_write_tensor(r_msb, p0_ptr, p1_ptr);
    }

    // For test_truncate_zero_extend_tensor_3d
    {
        FixTensor<T, M_BITS, F, K, 3> r_m(2, 2, 2); 
        FixTensor<T, BW, F, K, 3> r_e(2, 2, 2);
        FixTensor<T, BW, F, K, 3> r_msb(2, 2, 2);

        for (int i = 0; i < r_m.size(); ++i) {
            T val = rg.template randomGE<T>(1, M_BITS)[0];
            r_m.data()[i] = Fix<T, M_BITS, F, K>(val);
            r_e.data()[i] = Fix<T, BW, F, K>(val);
            r_msb.data()[i] = r_m.data()[i].template get_msb<BW, F, K>();
        }
        
        secret_share_and_write_tensor(r_m, p0_ptr, p1_ptr);
        secret_share_and_write_tensor(r_e, p0_ptr, p1_ptr);
        secret_share_and_write_tensor(r_msb, p0_ptr, p1_ptr);
    }

    // For test_elementwise_mul_opt
    {
        using FixTensorN = FixTensor<uint64_t, BW, F, K, 2>;
        using FixTensorM = FixTensor<uint64_t, M_BITS, F, K, 2>;
        const int D1 = 3, D2 = 4;

        FixTensorM R_X(D1, D2), R_Y(D1, D2);
        FixTensorN R_X_N(D1, D2), R_Y_N(D1, D2);
        FixTensorN R_X_MSB(D1, D2), R_Y_MSB(D1, D2);
        FixTensorN R_XY(D1, D2), R_X_RYMSB(D1, D2), R_XMSB_Y(D1, D2), R_XMSB_YMSB(D1, D2);
        
        Random rg;
        for(long long i = 0; i < R_X.size(); ++i) {
            T val = rg.template randomGE<T>(1, M_BITS)[0];
            R_X.data()[i] = Fix<T, M_BITS, F, K>(val);
            R_X_N.data()[i] = Fix<T, BW, F, K>(val);
            R_X_MSB.data()[i] = R_X.data()[i].template get_msb<BW, F, K>();
            val = rg.template randomGE<T>(1, M_BITS)[0];
            R_Y.data()[i] = Fix<T, M_BITS, F, K>(val);
            R_Y_N.data()[i] = Fix<T, BW, F, K>(val);
            R_Y_MSB.data()[i] = R_Y.data()[i].template get_msb<BW, F, K>();
        }
        R_XY = (R_X_N * R_Y_N);
        // for(long long i = 0; i < R_XY.size(); ++i){
        //     R_XY.data()[i].val = R_XY.data()[i].val & ((1ULL << M_BITS) - 1);
        // }
        R_X_RYMSB = R_X_N * R_Y_MSB;
        R_XMSB_Y = R_X_MSB * R_Y_N;
        R_XMSB_YMSB = R_X_MSB * R_Y_MSB;

        secret_share_and_write_tensor(R_X, p0_ptr, p1_ptr);
        secret_share_and_write_tensor(R_Y, p0_ptr, p1_ptr);
        secret_share_and_write_tensor(R_X_N, p0_ptr, p1_ptr);
        secret_share_and_write_tensor(R_Y_N, p0_ptr, p1_ptr);
        secret_share_and_write_tensor(R_X_MSB, p0_ptr, p1_ptr);
        secret_share_and_write_tensor(R_Y_MSB, p0_ptr, p1_ptr);
        secret_share_and_write_tensor(R_XY, p0_ptr, p1_ptr);
        secret_share_and_write_tensor(R_X_RYMSB, p0_ptr, p1_ptr);
        secret_share_and_write_tensor(R_XMSB_Y, p0_ptr, p1_ptr);
        secret_share_and_write_tensor(R_XMSB_YMSB, p0_ptr, p1_ptr);
    }

    // // For FC Layer Test
    // {
    //     fc_layer.dealer_generate_forward_randomness(p0_ptr, p1_ptr, U, V);
    //     fc_layer.dealer_generate_backward_randomness(p0_ptr, p1_ptr);
    // }

    // 4. Assert that we wrote the exact calculated size
    size_t p0_offset = p0_ptr - p0_data;
    size_t p1_offset = p1_ptr - p1_data;
    assert(p0_offset == total_size);
    assert(p1_offset == total_size);

    // 5. Write buffers to files
    std::ofstream p0_file("./randomness/P0/random_data.bin", std::ios::binary);
    p0_file.write(reinterpret_cast<const char*>(p0_data), total_size);
    p0_file.close();

    std::ofstream p1_file("./randomness/P1/random_data.bin", std::ios::binary);
    p1_file.write(reinterpret_cast<const char*>(p1_data), total_size);
    p1_file.close();

    // 6. Free memory
    delete[] p0_data;
    delete[] p1_data;

    std::cout << "Dealer finished successfully." << std::endl;
    return 0;
}
