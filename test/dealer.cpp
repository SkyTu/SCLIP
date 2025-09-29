#include <iostream>
#include <fstream>
#include <vector>
#include <cassert>
#include "mpc/fix_tensor.h"
#include "utils/random.h"
#include "mpc/tensor_ops.h"

// Secret share helper for Tensors
template<typename FixTensorType>
std::pair<FixTensorType, FixTensorType> secret_share_into_two(const FixTensorType& plaintext) {
    FixTensorType share0(plaintext.dimensions());
    FixTensorType share1(plaintext.dimensions());
    Random rg;

    for (long long i = 0; i < plaintext.size(); ++i) {
        using T = typename FixTensorType::Scalar::val_type;
        constexpr int bw = FixTensorType::Scalar::bitwidth;
        T r_val = rg.template randomGE<T>(1, bw)[0];
        share1.data()[i] = typename FixTensorType::Scalar(r_val);
        share0.data()[i] = plaintext.data()[i] - share1.data()[i];
    }
    return {share0, share1};
}

// Secret share helper for scalar Fix
template<typename FixType>
std::pair<FixType, FixType> secret_share_into_two_scalar(const FixType& plaintext) {
    using T = typename FixType::val_type;
    constexpr int bw = FixType::bitwidth;
    Random rg;
    T r_val = rg.template randomGE<T>(1, bw)[0];
    FixType share1(r_val);
    FixType share0 = plaintext - share1;
    return {share0, share1};
}

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
    
    std::cout << "Total size per party: " << total_size << " bytes." << std::endl;

    // 2. Allocate raw uint8_t* buffers
    uint8_t* p0_data = new uint8_t[total_size];
    uint8_t* p1_data = new uint8_t[total_size];
    size_t p0_offset = 0;
    size_t p1_offset = 0;

    // Helper lambda for writing shares to buffers
    auto write_shares_to_buffers = [&](const auto& tensor) {
        auto [share0, share1] = secret_share_into_two(tensor);
        size_t size_bytes = tensor.size() * sizeof(typename std::remove_reference_t<decltype(tensor)>::Scalar);
        
        memcpy(p0_data + p0_offset, share0.data(), size_bytes);
        p0_offset += size_bytes;
        
        memcpy(p1_data + p1_offset, share1.data(), size_bytes);
        p1_offset += size_bytes;
    };

    auto write_scalar_shares_to_buffers = [&](const auto& scalar) {
        auto [share0, share1] = secret_share_into_two_scalar(scalar);
        size_t size_bytes = sizeof(typename std::remove_reference_t<decltype(scalar)>::val_type);

        memcpy(p0_data + p0_offset, &share0.val, size_bytes);
        p0_offset += size_bytes;

        memcpy(p1_data + p1_offset, &share1.val, size_bytes);
        p1_offset += size_bytes;
    };

    // 3. Generate randomness and write to buffers
    // For test_secure_matmul (2D)
    {
        FixTensor<T, BW, F, K, 2> U(2, 3); U.initialize();
        FixTensor<T, BW, F, K, 2> V(3, 2); V.initialize();
        FixTensor<T, BW, F, K, 2> Z = tensor_mul(U, V);
        write_shares_to_buffers(U);
        write_shares_to_buffers(V);
        write_shares_to_buffers(Z);
    }
    
    // For test_secure_matmul_3d
    {
        FixTensor<T, BW, F, K, 3> U(2, 2, 3); U.initialize();
        FixTensor<T, BW, F, K, 2> V(3, 2); V.initialize();
        FixTensor<T, BW, F, K, 3> Z = tensor_mul(U, V);
        write_shares_to_buffers(U);
        write_shares_to_buffers(V);
        write_shares_to_buffers(Z);
    }

    // For test_truncate_zero_extend_scalar
    {
        T r_m_val = rg.template randomGE<T>(1, M_BITS)[0];
        Fix<T, M_BITS, F, K> r_m(r_m_val);
        Fix<T, BW, F, K> r_e = r_m.template change_format<BW, F, K>(); // r_e is the zero-extension of r_m
        Fix<T, BW, F, K> r_msb = r_m.template get_msb<BW, F, K>(); // r_msb is the MSB of r_m
        write_scalar_shares_to_buffers(r_m);
        write_scalar_shares_to_buffers(r_e);
        write_scalar_shares_to_buffers(r_msb);
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
        write_shares_to_buffers(r_m);
        write_shares_to_buffers(r_e);
        write_shares_to_buffers(r_msb);
    }

    // For test_truncate_zero_extend_tensor_3d
    {
        FixTensor<T, M_BITS, F, K, 3> r_m(2, 2, 2); 
        FixTensor<T, BW, F, K, 3> r_e(2, 2, 2);
        FixTensor<T, BW, F, K, 3> r_msb(2, 2, 2);

        for (int i = 0; i < r_m.size(); ++i) {
            T val = rg.template randomGE<T>(1, M_BITS)[0];
            r_m.data()[i] = Fix<T, M_BITS, F, K>(val);
            r_e.data()[i] = r_m.data()[i];
            r_msb.data()[i] = r_m.data()[i].template get_msb<BW, F, K>();
        }
        
        write_shares_to_buffers(r_m);
        write_shares_to_buffers(r_e);
        write_shares_to_buffers(r_msb);
    }

    // For test_elementwise_mul_opt
    {
        using FixTensorN = FixTensor<uint64_t, BW, F, K, 2>;
        using FixTensorM = FixTensor<uint64_t, M_BITS, F, K, 2>;
        const int D1 = 3, D2 = 4;

        FixTensorM R_X(D1, D2), R_Y(D1, D2);
        
        Random rg;
        uint64_t* r_x_data = rg.template randomGE<uint64_t>(R_X.size(), M_BITS);
        for(long long i = 0; i < R_X.size(); ++i) {
            R_X.data()[i] = Fix<uint64_t, M_BITS, F, K>(r_x_data[i]);
        }
        delete[] r_x_data;

        uint64_t* r_y_data = rg.template randomGE<uint64_t>(R_Y.size(), M_BITS);
        for(long long i = 0; i < R_Y.size(); ++i) {
            R_Y.data()[i] = Fix<uint64_t, M_BITS, F, K>(r_y_data[i]);
        }
        delete[] r_y_data;


        
        FixTensorN R_X_MSB = get_msb<BW, F, K>(R_X);
        FixTensorN R_Y_MSB = get_msb<BW, F, K>(R_Y);
        FixTensorN R_X_N = extend_locally<BW, F, K>(R_X);
        FixTensorN R_Y_N = extend_locally<BW, F, K>(R_Y);
        FixTensorN R_XY = R_X_N * R_Y_N;
        FixTensorN R_X_RYMSB = R_X_N * R_Y_MSB;
        FixTensorN R_XMSB_Y = R_X_MSB * R_Y_N;
        FixTensorN R_XMSB_YMSB = R_X_MSB * R_Y_MSB;

        write_shares_to_buffers(R_X);
        write_shares_to_buffers(R_Y);
        write_shares_to_buffers(R_X_N);
        write_shares_to_buffers(R_Y_N);
        write_shares_to_buffers(R_X_MSB);
        write_shares_to_buffers(R_Y_MSB);
        write_shares_to_buffers(R_XY);
        write_shares_to_buffers(R_X_RYMSB);
        write_shares_to_buffers(R_XMSB_Y);
        write_shares_to_buffers(R_XMSB_YMSB);
    }

    // 4. Assert that we wrote the exact calculated size
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
