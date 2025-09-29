#include "mpc/fix_tensor.h"
#include "utils/random.h"
#include "utils/compress.h"
#include "mpc/tensor_ops.h"
#include <iostream>
#include <vector>
#include <string>
#include <fstream>
#include <numeric>

// =================================================================================
// 1. Size Calculation Functions
// =================================================================================

// Get the size of a Fix share in bytes.
template <typename FixType>
size_t get_fix_share_size_in_bytes() {
    return (FixType::bitwidth + 7) / 8;
}

// Get the size of a FixTensor share in bytes (dimensions + data).
template <typename FixTensorType>
size_t get_tensor_share_size_in_bytes(const FixTensorType& tensor_template) {
    size_t dim_bytes = FixTensorType::NumDimensions * sizeof(long);
    size_t data_bytes = (tensor_template.size() * FixTensorType::FixType::bitwidth + 7) / 8;
    return dim_bytes + data_bytes;
}

// =================================================================================
// 2. Data Writing Functions
// =================================================================================

// Write a Fix share to the buffer and advance the offset.
template <typename FixType>
void write_fix_share(uint8_t* buffer, size_t& offset, const FixType& share) {
    std::vector<uint8_t> packed_share = pack_data(&share.val, 1, FixType::bitwidth);
    memcpy(buffer + offset, packed_share.data(), packed_share.size());
    offset += packed_share.size();
}

// Write a FixTensor share to the buffer and advance the offset.
template <typename FixTensorType>
void write_tensor_share(uint8_t* buffer, size_t& offset, const FixTensorType& share) {
    // Write dimensions
    for (long d : share.dimensions()) {
        memcpy(buffer + offset, &d, sizeof(long));
        offset += sizeof(long);
    }
    // Write data
    std::vector<uint8_t> packed_data = pack_data(
        reinterpret_cast<const typename FixTensorType::FixType::val_type*>(share.data()),
        share.size(),
        FixTensorType::FixType::bitwidth
    );
    memcpy(buffer + offset, packed_data.data(), packed_data.size());
    offset += packed_data.size();
}

// =================================================================================
// 3. Secret Sharing Helpers (from previous attempts, now memory-safe)
// =================================================================================

template<typename FixType>
std::pair<FixType, FixType> secret_share_into_two_scalar(const FixType& plaintext, Random& rg) {
    using T = typename FixType::val_type;
    T* r_ptr = rg.randomGE<T>(1, FixType::bitwidth);
    FixType share1(r_ptr[0]);
    delete[] r_ptr;
    FixType share0 = plaintext - share1;
    return {share0, share1};
}

template<typename FixTensorType>
std::pair<FixTensorType, FixTensorType> secret_share_into_two_tensor(const FixTensorType& plaintext, Random& rg) {
    using T = typename FixTensorType::FixType::val_type;
    FixTensorType share1(plaintext.dimensions());
    T* r_ptr = rg.randomGE<T>(plaintext.size(), FixTensorType::FixType::bitwidth);
    memcpy(share1.data(), r_ptr, plaintext.size() * sizeof(typename FixTensorType::FixType));
    delete[] r_ptr;
    FixTensorType share0 = plaintext - share1;
    return {share0, share1};
}


// =================================================================================
// 4. Main Dealer Logic
// =================================================================================

int main() {
    std::cout << "Dealer starting..." << std::endl;

    // Create directories
    system("mkdir -p ./randomness/P0");
    system("mkdir -p ./randomness/P1");

    // Define types and dimensions
    using T = uint64_t;
    const int BW = 64;
    const int F = 16;
    const int K = 31;
    const int M_BITS = BW - F;

    using FixM = Fix<T, M_BITS, F, K>;
    using FixBW = Fix<T, BW, F, K>;

    // Matmul 2D templates
    FixTensor<T, BW, F, K, 2> mat2d_U_template(2, 3);
    FixTensor<T, BW, F, K, 2> mat2d_V_template(3, 2);
    FixTensor<T, BW, F, K, 2> mat2d_Z_template(2, 2);

    // Matmul 3D templates
    FixTensor<T, BW, F, K, 3> mat3d_U_template(2, 2, 3);
    FixTensor<T, BW, F, K, 2> mat3d_V_template(3, 2);
    FixTensor<T, BW, F, K, 3> mat3d_Z_template(2, 2, 2);
    
    // ZE Tensor templates
    FixTensor<T, M_BITS, F, K, 2> ze2d_R_template(2, 2);
    FixTensor<T, BW, F, K, 2>    ze2d_RE_template(2, 2);
    FixTensor<T, BW, F, K, 2>    ze2d_RMSB_template(2, 2);
    FixTensor<T, M_BITS, F, K, 3> ze3d_R_template(2, 2, 2);
    FixTensor<T, BW, F, K, 3>    ze3d_RE_template(2, 2, 2);
    FixTensor<T, BW, F, K, 3>    ze3d_RMSB_template(2, 2, 2);

    // Elementwise Mul Opt templates
    using FixTensorM = FixTensor<T, M_BITS, F, K, 2>;
    using FixTensorN = FixTensor<T, BW, F, K, 2>;
    FixTensorM ele_rx_m_template(2, 2);
    FixTensorN ele_rx_n_template(2, 2);
    FixTensorN ele_rx_msb_n_template(2, 2);
    FixTensorM ele_ry_m_template(2, 2);
    FixTensorN ele_ry_n_template(2, 2);
    FixTensorN ele_ry_msb_n_template(2, 2);
    FixTensorN ele_rxy_n_template(2, 2);
    FixTensorN ele_rx_msby_n_template(2, 2);
    FixTensorN ele_rxy_msb_n_template(2, 2);
    FixTensorN ele_rx_msby_msb_n_template(2, 2);

    // Calculate total size
    size_t total_size = 0;
    total_size += get_tensor_share_size_in_bytes(mat2d_U_template);
    total_size += get_tensor_share_size_in_bytes(mat2d_V_template);
    total_size += get_tensor_share_size_in_bytes(mat2d_Z_template);
    total_size += get_tensor_share_size_in_bytes(mat3d_U_template);
    total_size += get_tensor_share_size_in_bytes(mat3d_V_template);
    total_size += get_tensor_share_size_in_bytes(mat3d_Z_template);
    total_size += get_fix_share_size_in_bytes<FixM>();    // ze_s_r
    total_size += get_fix_share_size_in_bytes<FixBW>();   // ze_s_re
    total_size += get_fix_share_size_in_bytes<FixBW>();   // ze_s_rmsb
    total_size += get_tensor_share_size_in_bytes(ze2d_R_template);
    total_size += get_tensor_share_size_in_bytes(ze2d_RE_template);
    total_size += get_tensor_share_size_in_bytes(ze2d_RMSB_template);
    total_size += get_tensor_share_size_in_bytes(ze3d_R_template);
    total_size += get_tensor_share_size_in_bytes(ze3d_RE_template);
    total_size += get_tensor_share_size_in_bytes(ze3d_RMSB_template);
    total_size += get_tensor_share_size_in_bytes(ele_rx_m_template);
    total_size += get_tensor_share_size_in_bytes(ele_rx_n_template);
    total_size += get_tensor_share_size_in_bytes(ele_rx_msb_n_template);
    total_size += get_tensor_share_size_in_bytes(ele_ry_m_template);
    total_size += get_tensor_share_size_in_bytes(ele_ry_n_template);
    total_size += get_tensor_share_size_in_bytes(ele_ry_msb_n_template);
    total_size += get_tensor_share_size_in_bytes(ele_rxy_n_template);
    total_size += get_tensor_share_size_in_bytes(ele_rx_msby_n_template);
    total_size += get_tensor_share_size_in_bytes(ele_rxy_msb_n_template);
    total_size += get_tensor_share_size_in_bytes(ele_rx_msby_msb_n_template);
    
    std::cout << "Total size per party: " << total_size << " bytes." << std::endl;

    // Allocate buffers
    uint8_t* p0_shares = new uint8_t[total_size];
    uint8_t* p1_shares = new uint8_t[total_size];
    size_t p0_offset = 0;
    size_t p1_offset = 0;

    Random rg;

    // === Generate and Write Matmul 2D ===
    auto mat2d_U = mat2d_U_template; mat2d_U.initialize();
    auto mat2d_V = mat2d_V_template; mat2d_V.initialize();
    auto mat2d_Z = tensor_mul(mat2d_U, mat2d_V);
    auto u_shares = secret_share_into_two_tensor(mat2d_U, rg);
    auto v_shares = secret_share_into_two_tensor(mat2d_V, rg);
    auto z_shares = secret_share_into_two_tensor(mat2d_Z, rg);
    write_tensor_share(p0_shares, p0_offset, u_shares.first);
    write_tensor_share(p1_shares, p1_offset, u_shares.second);
    write_tensor_share(p0_shares, p0_offset, v_shares.first);
    write_tensor_share(p1_shares, p1_offset, v_shares.second);
    write_tensor_share(p0_shares, p0_offset, z_shares.first);
    write_tensor_share(p1_shares, p1_offset, z_shares.second);

    // === Generate and Write Matmul 3D ===
    auto mat3d_U = mat3d_U_template; mat3d_U.initialize();
    auto mat3d_V = mat3d_V_template; mat3d_V.initialize();
    auto mat3d_Z = tensor_mul(mat3d_U, mat3d_V);
    auto u3_shares = secret_share_into_two_tensor(mat3d_U, rg);
    auto v3_shares = secret_share_into_two_tensor(mat3d_V, rg);
    auto z3_shares = secret_share_into_two_tensor(mat3d_Z, rg);
    write_tensor_share(p0_shares, p0_offset, u3_shares.first);
    write_tensor_share(p1_shares, p1_offset, u3_shares.second);
    write_tensor_share(p0_shares, p0_offset, v3_shares.first);
    write_tensor_share(p1_shares, p1_offset, v3_shares.second);
    write_tensor_share(p0_shares, p0_offset, z3_shares.first);
    write_tensor_share(p1_shares, p1_offset, z3_shares.second);

    // === Generate and Write Zero-Extend Triples ===
    T m_mask = (M_BITS == 64) ? ~T(0) : ((T(1) << M_BITS) - 1);
    
    // Scalar
    FixM r_s(rg.randomGE<T>(1, M_BITS)[0] & m_mask);
    FixBW r_e_s(r_s.val);
    FixBW r_msb_s((r_s.val >> (M_BITS - 1)) & 1);
    auto r_s_shares = secret_share_into_two_scalar(r_s, rg);
    auto r_e_s_shares = secret_share_into_two_scalar(r_e_s, rg);
    auto r_msb_s_shares = secret_share_into_two_scalar(r_msb_s, rg);
    write_fix_share(p0_shares, p0_offset, r_s_shares.first);
    write_fix_share(p1_shares, p1_offset, r_s_shares.second);
    write_fix_share(p0_shares, p0_offset, r_e_s_shares.first);
    write_fix_share(p1_shares, p1_offset, r_e_s_shares.second);
    write_fix_share(p0_shares, p0_offset, r_msb_s_shares.first);
    write_fix_share(p1_shares, p1_offset, r_msb_s_shares.second);

    // 2D
    auto r_2d = ze2d_R_template;
    auto r_e_2d = ze2d_RE_template;
    auto r_msb_2d = ze2d_RMSB_template;
    for(long i = 0; i < r_2d.size(); ++i) {
        T r_val = rg.randomGE<T>(1, M_BITS)[0] & m_mask;
        r_2d.data()[i] = FixM(r_val);
        r_e_2d.data()[i] = FixBW(r_val);
        r_msb_2d.data()[i] = FixBW((r_val >> (M_BITS - 1)) & 1);
    }
    auto r2_shares = secret_share_into_two_tensor(r_2d, rg);
    auto re2_shares = secret_share_into_two_tensor(r_e_2d, rg);
    auto rmsb2_shares = secret_share_into_two_tensor(r_msb_2d, rg);
    write_tensor_share(p0_shares, p0_offset, r2_shares.first);
    write_tensor_share(p1_shares, p1_offset, r2_shares.second);
    write_tensor_share(p0_shares, p0_offset, re2_shares.first);
    write_tensor_share(p1_shares, p1_offset, re2_shares.second);
    write_tensor_share(p0_shares, p0_offset, rmsb2_shares.first);
    write_tensor_share(p1_shares, p1_offset, rmsb2_shares.second);

    // 3D
    auto r_3d = ze3d_R_template;
    auto r_e_3d = ze3d_RE_template;
    auto r_msb_3d = ze3d_RMSB_template;
    for(long i = 0; i < r_3d.size(); ++i) {
        T r_val = rg.randomGE<T>(1, M_BITS)[0] & m_mask;
        r_3d.data()[i] = FixM(r_val);
        r_e_3d.data()[i] = FixBW(r_val);
        r_msb_3d.data()[i] = FixBW((r_val >> (M_BITS - 1)) & 1);
    }
    auto r3_shares = secret_share_into_two_tensor(r_3d, rg);
    auto re3_shares = secret_share_into_two_tensor(r_e_3d, rg);
    auto rmsb3_shares = secret_share_into_two_tensor(r_msb_3d, rg);
    write_tensor_share(p0_shares, p0_offset, r3_shares.first);
    write_tensor_share(p1_shares, p1_offset, r3_shares.second);
    write_tensor_share(p0_shares, p0_offset, re3_shares.first);
    write_tensor_share(p1_shares, p1_offset, re3_shares.second);
    write_tensor_share(p0_shares, p0_offset, rmsb3_shares.first);
    write_tensor_share(p1_shares, p1_offset, rmsb3_shares.second);

    // Final check
    std::cout << "Final P0 offset: " << p0_offset << " (Total size: " << total_size << ")" << std::endl;
    std::cout << "Final P1 offset: " << p1_offset << " (Total size: " << total_size << ")" << std::endl;
    assert(p0_offset == total_size);
    assert(p1_offset == total_size);

    // Write to files
    std::ofstream p0_file("./randomness/P0/random_data.bin", std::ios::binary);
    p0_file.write(reinterpret_cast<const char*>(p0_shares), total_size);
    p0_file.close();

    std::ofstream p1_file("./randomness/P1/random_data.bin", std::ios::binary);
    p1_file.write(reinterpret_cast<const char*>(p1_shares), total_size);
    p1_file.close();
    
    // Cleanup
    delete[] p0_shares;
    delete[] p1_shares;

    std::cout << "Dealer finished generating randomness." << std::endl;
    return 0;
}
