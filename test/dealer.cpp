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
    total_size += get_matmul_random_size<T, BW, F, K, 2, 2, 2>(2, 2, 3);
    // For test_secure_matmul_3d
    total_size += get_matmul_random_size<T, BW, F, K, 3, 2, 3>(2, 2, 3, 2);
    // For test_truncate_zero_extend_scalar
    total_size += 3 * scalar_size;
    // For test_truncate_zero_extend_tensor_2d
    total_size += get_zero_extend_random_size<T, M_BITS, BW, F, K, 2>(20, 20, 20);
    // For test_truncate_zero_extend_tensor_3d
    total_size += get_zero_extend_random_size<T, M_BITS, BW, F, K, 3>(2, 2, 2);
    // For test_elementwise_mul_opt
    total_size += get_elementwise_mul_random_size<T, M_BITS, F, K, BW, 2, 2>(3, 4, 4);
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
        generate_matmul_randomness<T, BW, F, K, 2, 2, 2>(p0_ptr, p1_ptr, 2, 2, 3);
    }
    
    // For test_secure_matmul_3d
    {
        generate_matmul_randomness<T, BW, F, K, 3, 2, 3>(p0_ptr, p1_ptr, 2, 2, 3, 2);
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
        generate_zero_extend_randomness<T, M_BITS, BW, F, K, 2>(20, 20, 20, p0_ptr, p1_ptr);
    }

    // For test_truncate_zero_extend_tensor_3d
    {
        generate_zero_extend_randomness<T, M_BITS, BW, F, K, 3>(2, 2, 2, p0_ptr, p1_ptr);
    }

    // For test_elementwise_mul_opt
    {
        generate_elementwise_mul_randomness<T, M_BITS, F, K, BW, 2, 2>(3, 4, 4, p0_ptr, p1_ptr);
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
