#include <iostream>
#include <fstream>
#include <vector>
#include <cassert>
#include "mpc/fix_tensor.h"
#include "utils/random.h"
#include "mpc/tensor_ops.h"
#include "mpc/matmul.h"
#include "nn/FC.h"
#include "mpc/square.h"
#include "mpc/elementwise_mul.h"
#include "mpc/secure_tensor_ops.h"



int main() {
    std::cout << "Dealer starting..." << std::endl;
    system("mkdir -p ./randomness/P0 && mkdir -p ./randomness/P1");

    using T = uint64_t;
    const int BW = 64;
    const int F = 16;
    const int K = 31;
    constexpr int M_BITS = BW - F;
    Random rg;

    // 1. Pre-calculate total size
    size_t total_size = 0;
    const size_t scalar_size = sizeof(T);
    // For test_secure_matmul (2D)
    total_size += get_matmul_random_size<T, BW, F, K, 2, 2, 2>(2, 3, 2);
    std::cout << "total_size: " << total_size << std::endl;
    // For test_secure_matmul (3D x 2D)
    total_size += get_matmul_random_size<T, BW, F, K, 3, 2, 3>(2, 2, 3, 2);  
    std::cout << "total_size: " << total_size << std::endl;
    // // For test_truncate_zero_extend_scalar
    // total_size += 10 * 3 * scalar_size;
    // std::cout << "total_size: " << total_size << std::endl;
    // For test_truncate_zero_extend_tensor_2d
    total_size += get_zero_extend_random_size<T, 2>(20, 20, 20);
    std::cout << "total_size: " << total_size << std::endl;
    // For test_truncate_zero_extend_tensor_3d
    total_size += get_zero_extend_random_size<T, 3>(2, 2, 2);
    std::cout << "total_size: " << total_size << std::endl;
    // For test_elementwise_mul_opt
    total_size += get_elementwise_mul_random_size<T, 2>(0, 3, 4);
    std::cout << "total_size: " << total_size << std::endl;
    // For test_square_tensor_opt
    total_size += get_square_random_size<T, 2>(0, 3, 4);
    std::cout << "total_size: " << total_size << std::endl;
    // For test_square_tensor_opt_3d
    total_size += get_square_random_size<T, 3>(20, 20, 20);
    std::cout << "total_size: " << total_size << std::endl;
    // // for test square scalar opt
    // total_size += 10 * get_square_scalar_random_size<T>();
    // std::cout << "total_size: " << total_size << std::endl;
    // // For test_exp_scalar_opt
    // total_size += get_exp_scalar_random_size<T>();
    // // For test_exp_tensor_opt_3d
    total_size += get_exp_random_size<T, 3>(20, 20, 20);
    std::cout << "Total size per party: " << total_size << " bytes." << std::endl;
    // For test_inv_sqrt_tensor
    total_size += get_inv_sqrt_random_size<T, BW, M_BITS, F, K, 3>(1, 3, 3);
    std::cout << "total_size: " << total_size << std::endl;

    // 2. Allocate raw uint8_t* buffers
    uint8_t* p0_data = new uint8_t[total_size];
    uint8_t* p1_data = new uint8_t[total_size];

    // 3. Generate randomness and write to buffers
    Buffer p0_buf(p0_data);
    Buffer p1_buf(p1_data);

    // For test_secure_matmul (2D)
    {
        generate_matmul_randomness<T, BW, F, K, 2, 2, 2>(p0_buf, p1_buf, 0, 2, 3, 2);
    }
    
    size_t p0_offset = p0_buf.ptr - p0_data;
    size_t p1_offset = p1_buf.ptr - p1_data;
    std::cout << "p0_offset: " << p0_offset << ", p1_offset: " << p1_offset << ", total_size: " << total_size << std::endl;
    
    // For test_secure_matmul_3d
    {
        generate_matmul_randomness<T, BW, F, K, 3, 2, 3>(p0_buf, p1_buf, 2, 2, 3, 2);
    }

    // p0_offset = p0_buf.ptr - p0_data;
    // p1_offset = p1_buf.ptr - p1_data;
    // std::cout << "p0_offset: " << p0_offset << ", p1_offset: " << p1_offset << ", total_size: " << total_size << std::endl;

    // // For test_truncate_zero_extend_scalar
    // {
    //     for (int i = 0; i < 10; i++) {
    //         T r_m_val = rg.template randomGE<T>(1, M_BITS)[0];
    //         Fix<T, M_BITS, F, K> r_m(r_m_val);
    //         Fix<T, BW, F, K> r_e(r_m_val); // r_e is the zero-extension of r_m
    //         Fix<T, BW, F, K> r_msb = r_m.template get_msb<BW, F, K>(); // r_msb is the MSB of r_m
    //         secret_share_and_write_scalar<Fix<T, M_BITS, F, K>>(r_m, p0_buf, p1_buf);
    //         secret_share_and_write_scalar<Fix<T, BW, F, K>>(r_e, p0_buf, p1_buf);
    //         secret_share_and_write_scalar<Fix<T, BW, F, K>>(r_msb, p0_buf, p1_buf);
    //     }
    // }
    
    p0_offset = p0_buf.ptr - p0_data;
    p1_offset = p1_buf.ptr - p1_data;
    std::cout << "p0_offset: " << p0_offset << ", p1_offset: " << p1_offset << ", total_size: " << total_size << std::endl;
    // For test_truncate_zero_extend_tensor_2d
    {
        generate_zero_extend_randomness<T, BW, M_BITS, F, K, 2>(20, 20, 20, p0_buf, p1_buf);
    }

    p0_offset = p0_buf.ptr - p0_data;
    p1_offset = p1_buf.ptr - p1_data;
    std::cout << "p0_offset: " << p0_offset << ", p1_offset: " << p1_offset << ", total_size: " << total_size << std::endl;
    // For test_truncate_zero_extend_tensor_3d
    {
        generate_zero_extend_randomness<T, BW, M_BITS, F, K, 3>(2, 2, 2, p0_buf, p1_buf);
    }

    p0_offset = p0_buf.ptr - p0_data;
    p1_offset = p1_buf.ptr - p1_data;
    std::cout << "p0_offset: " << p0_offset << ", p1_offset: " << p1_offset << ", total_size: " << total_size << std::endl;
    // For test_elementwise_mul_opt
    {
        generate_elementwise_mul_randomness<T, BW, M_BITS, F, K, 2, Eigen::RowMajor>(0, 3, 4, p0_buf, p1_buf);
    }

    p0_offset = p0_buf.ptr - p0_data;
    p1_offset = p1_buf.ptr - p1_data;
    std::cout << "p0_offset: " << p0_offset << ", p1_offset: " << p1_offset << ", total_size: " << total_size << std::endl;
    // For test_square_tensor_opt
    {
        generate_square_randomness<T, BW, M_BITS, F, K, 2>(0, 3, 4, p0_buf, p1_buf);
    }

    p0_offset = p0_buf.ptr - p0_data;
    p1_offset = p1_buf.ptr - p1_data;
    std::cout << "p0_offset: " << p0_offset << ", p1_offset: " << p1_offset << ", total_size: " << total_size << std::endl;
    // For test_square_tensor_opt_3d
    {
        generate_square_randomness<T, BW, M_BITS, F, K, 3>(20, 20, 20, p0_buf, p1_buf);
    }

    // p0_offset = p0_buf.ptr - p0_data;
    // p1_offset = p1_buf.ptr - p1_data;
    // std::cout << "p0_offset: " << p0_offset << ", p1_offset: " << p1_offset << ", total_size: " << total_size << std::endl;
    // // For test square scalar opt
    // {
    //     for(int i = 0; i < 10; i++){
    //         generate_square_scalar_randomness<T, BW, M_BITS, F, K>(p0_buf, p1_buf);
    //     }
    // }

    // p0_offset = p0_buf.ptr - p0_data;
    // p1_offset = p1_buf.ptr - p1_data;
    // std::cout << "p0_offset: " << p0_offset << ", p1_offset: " << p1_offset << ", total_size: " << total_size << std::endl;
    // // For test_exp_scalar_opt
    // {
    //     generate_exp_scalar_randomness<T, BW, M_BITS, F, K>(p0_buf, p1_buf);
    // }

    // For test_exp_tensor_opt_3d
    {
        generate_exp_randomness<T, BW, M_BITS, F, K, 3>(20, 20, 20, p0_buf, p1_buf);
    }

    // For test_inverse_sqrt_3d
    {
        generate_inv_sqrt_randomness<T, BW, M_BITS, F, K, 3>(1, 3, 3, p0_buf, p1_buf);
    }

    // 4. Assert that we wrote the exact calculated size
    // p0_offset = p0_buf.ptr - p0_data;
    // p1_offset = p1_buf.ptr - p1_data;
    // std::cout << "p0_offset: " << p0_offset << ", p1_offset: " << p1_offset << ", total_size: " << total_size << std::endl;
    // assert(p0_offset == total_size);
    // assert(p1_offset == total_size);

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
