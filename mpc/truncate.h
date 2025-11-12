#pragma once

#include "mpc/fix.h"
#include "mpc/fix_tensor.h"
#include "mpc/tensor_ops.h"
#include "mpc/mpc.h"
#include "utils/random.h"
#include <iostream>

template <typename T, int BW, int smallBW, int F, int K, int Rank>
struct ZeroExtendRandomness{
    FixTensor<T, smallBW, F, K, Rank> R_M;
    FixTensor<T, BW, F, K, Rank> R_E;
    FixTensor<T, BW, F, K, Rank> R_MSB;
};

template <typename T, int BW, int smallBW, int F, int K>
struct ZeroExtendScalarRandomness{
    Fix<T, smallBW, F, K> R;
    Fix<T, BW, F, K> R_E;
    Fix<T, BW, F, K> R_MSB;
};

template <typename T, int BW, int smallBW, int F, int K>
ZeroExtendScalarRandomness<T, BW, smallBW, F, K> read_zero_extend_scalar_randomness(MPC& mpc){
    ZeroExtendScalarRandomness<T, BW, smallBW, F, K> randomness;
    mpc.read_fix_share(randomness.R);
    mpc.read_fix_share(randomness.R_E);
    mpc.read_fix_share(randomness.R_MSB);
    return randomness;
}

template <typename T, int BW, int smallBW, int F, int K, int Rank>
ZeroExtendRandomness<T, BW, smallBW, F, K, Rank> read_zero_extend_randomness(MPC& mpc, int batch, int row, int col){
    ZeroExtendRandomness<T, BW, smallBW, F, K, Rank> randomness;
    if constexpr(Rank == 3){
        randomness.R_M.resize(batch, row, col);
        randomness.R_E.resize(batch, row, col);
        randomness.R_MSB.resize(batch, row, col);
    }
    else if constexpr(Rank == 2){
        randomness.R_M.resize(row, col);
        randomness.R_E.resize(row, col);
        randomness.R_MSB.resize(row, col);
    }
    else if constexpr(Rank == 1){
        randomness.R_M.resize(col);
        randomness.R_E.resize(col);
        randomness.R_MSB.resize(col);
    }
    else{
        throw std::runtime_error("Invalid rank for zero extend randomness");
    }
    mpc.read_fixtensor_share(randomness.R_M);
    mpc.read_fixtensor_share(randomness.R_E);
    mpc.read_fixtensor_share(randomness.R_MSB);
    return randomness;
}

template <typename T>
int get_zero_extend_scalar_random_size(){
    return 3 * sizeof(T);
}

template <typename T, int Rank>
int get_zero_extend_random_size(int batch, int row, int col){
    if(Rank == 3){
        return (batch * row * col + batch * row * col + batch * row * col) * sizeof(T);
    }else if (Rank == 2){
        return (row * col + row * col + row * col) * sizeof(T);
    }else if (Rank == 1){
        return (col + col + col) * sizeof(T);
    }
    else{
        throw std::runtime_error("Invalid rank for zero extend randomness");
    }
}

template <typename T, int BW, int smallBW, int F, int K>
void generate_zero_extend_scalar_randomness(Buffer& p0_buf, Buffer& p1_buf){
    Random rg;
    T r_val = rg.template randomGE<T>(1, smallBW)[0];
    Fix<T, smallBW, F, K> r_m = Fix<T, smallBW, F, K>(r_val);
    Fix<T, BW, F, K> r_e = Fix<T, BW, F, K>(r_val);
    Fix<T, BW, F, K> r_msb = r_m.template get_msb<BW, F, K>();
    secret_share_and_write_scalar<Fix<T, smallBW, F, K>>(r_m, p0_buf, p1_buf);
    secret_share_and_write_scalar<Fix<T, BW, F, K>>(r_e, p0_buf, p1_buf);
    secret_share_and_write_scalar<Fix<T, BW, F, K>>(r_msb, p0_buf, p1_buf);
}

template <typename T, int BW, int smallBW, int F, int K, int Rank>
void generate_zero_extend_randomness(Buffer& p0_buf, Buffer& p1_buf, int batch, int row, int col, FixTensor<T, smallBW, F, K, Rank>* r_m = nullptr){
    FixTensor<T, BW, F, K, Rank> r_e;
    FixTensor<T, BW, F, K, Rank> r_msb;
    Random rg;
    if constexpr (Rank == 3) {
        r_e.resize(batch, row, col);
        r_msb.resize(batch, row, col);
        T* val = rg.template randomGE<T>(batch * row * col, smallBW);
        if(r_m == nullptr){
            r_m = new FixTensor<T, smallBW, F, K, Rank>(batch, row, col);
            for (int i = 0; i < batch; i++) {
                for (int j = 0; j < row; j++) {
                    for (int z = 0; z < col; z++) {
                        (*r_m)(i, j, z) = Fix<T, smallBW, F, K>(val[i * row * col + j * col + z]);
                    }
                }
            }
        }
        r_e = extend_locally<BW, F, K>(*r_m);
        r_msb = get_msb<BW, F, K>(*r_m);    
        delete[] val; // <--- 必须加上这一行！
    } else if constexpr (Rank == 2) {
        r_e.resize(row, col);
        r_msb.resize(row, col);
        T* val = rg.template randomGE<T>(row * col, smallBW);
        if(r_m == nullptr){
            r_m = new FixTensor<T, smallBW, F, K, Rank>(row, col);
            for (int i = 0; i < row; i++) {
                for (int j = 0; j < col; j++) {
                    (*r_m)(i, j) = Fix<T, smallBW, F, K>(val[i * col + j]);
                }
            }
        }
        delete[] val; // <--- 必须加上这一行！
        r_e = extend_locally<BW, F, K>(*r_m);
        r_msb = get_msb<BW, F, K>(*r_m);
    }
    else{
        r_e.resize(col);
        r_msb.resize(col);
        T* val = rg.template randomGE<T>(col, smallBW);
        
        if(r_m == nullptr){
            r_m = new FixTensor<T, smallBW, F, K, Rank>(col);
            for (int i = 0; i < col; i++) {
                (*r_m)(i) = Fix<T, smallBW, F, K>(val[i]); 
            }
        }
        delete[] val; // <--- 必须加上这一行！
        r_e = extend_locally<BW, F, K>(*r_m);
        r_msb = get_msb<BW, F, K>(*r_m);
    }
    secret_share_and_write_tensor(*r_m, p0_buf, p1_buf);
    secret_share_and_write_tensor(r_e, p0_buf, p1_buf);
    secret_share_and_write_tensor(r_msb, p0_buf, p1_buf);
}

template <typename T, int bw, int f, int k>
Fix<T, (bw - f), f, k> truncate_reduce(const Fix<T, bw, f, k>& x_share) {
    static_assert(bw > f, "truncate_reduce requires bw > f");
    constexpr int m = bw - f;
    T new_val = static_cast<T>(x_share.val >> f);
    return Fix<T, m, f, k>(new_val);
}

template <typename T, int bw, int f, int k, int Rank, int Options>
FixTensor<T, (bw - f), f, k, Rank, Options>
truncate_reduce_tensor(const FixTensor<T, bw, f, k, Rank, Options>& x_share) {
    static_assert(bw > f, "truncate_reduce_tensor requires bw > f");
    constexpr int m = bw - f;
    FixTensor<T, m, f, k, Rank, Options> result(x_share.dimensions());
    for (long long i = 0; i < x_share.size(); ++i) {
        result.data()[i] = Fix<T, m, f, k>(static_cast<T>(x_share.data()[i].val >> f));
    }
    return result;
}

// ================= Zero Extend using extension triples (input m-bit ring -> output bw > m) =================

// Layout per element (per party): [r_m_share (m-ring), r_e_share (bw-ring), r_msb_share (bw-ring)]

template <typename T, int bw, int m, int f, int k>
Fix<T, bw, f, k> zero_extend(const Fix<T, m, f, k>& x_m_share, ZeroExtendScalarRandomness<T, bw, m, f, k> randomness) {
    static_assert(bw > m, "zero_extend requires bw > m");
    if (mpc_instance == nullptr) throw std::runtime_error("MPC instance not initialized.");
    
    const T bw_mask = (bw == 64) ? ~T(0) : ((T(1) << bw) - 1);

    int M = mpc_instance->M;
    int party = mpc_instance->party;
    Fix<T, m, f, k> bias = (party == 1 && m >= 2) ? Fix<T, m, f, k>(static_cast<T>(T(1) << (m - 2))) : Fix<T, m, f, k>(static_cast<T>(0));
    // 1) xhat_m_share = (x_m_share + r_m_share) in m-ring
    Fix<T, m, f, k> xhat_share_m = x_m_share + randomness.R + bias;

    // 2) Reconstruct xhat (public), then MSB at bit (m-1)
    Fix<T, m, f, k> xhat_m_public = reconstruct(xhat_share_m);
    T msb = (xhat_m_public.val >> (m - 1)) & T(1);

    // 3) e = 2^m * <r^msb>  (in bw-ring)
    T two_pow_m = (m >= 64) ? T(0) : (T(1) << m);
    T e_share_val = (two_pow_m == T(0)) ? T(0) : (randomness.R_MSB.val * two_pow_m) & bw_mask;

    // 4) t = <e> * (1 - MSB(xhat))
    T one_minus_msb = T(1) - msb;
    T t_share_val = (e_share_val * one_minus_msb) & bw_mask;

    // 5) Return σ·(x̂ − 2^{m−2}) − <r^e> + <t>  (mod 2^bw)
    T bias_val = (m >= 2) ? (T(1) << (m - 2)) : T(0);
    T term_sigma = (party == 1 ? ((xhat_m_public.val + bw_mask + 1) - bias_val) & bw_mask : T(0));
    T xext_val = (term_sigma + ((t_share_val + bw_mask + 1) - randomness.R_E.val)) & bw_mask;
    return Fix<T, bw, f, k>(xext_val);
}

// when the input is a reconstructed x_hat, not a share
template <typename T, int m, int f, int k, int bw>
Fix<T, bw, f, k> zero_extend_reconstructed(const Fix<T, m, f, k>& x_hat, const Fix<T, m, f, k>& r_m_share, const Fix<T, bw, f, k>& r_e_share, const Fix<T, bw, f, k>& r_msb_share) {
    static_assert(bw > m, "zero_extend requires bw > m");
    if (mpc_instance == nullptr) throw std::runtime_error("MPC instance not initialized.");
    int party = mpc_instance->party;
    const T bw_mask = (bw == 64) ? ~T(0) : ((T(1) << bw) - 1);
    Fix<T, m, f, k> bias = (m >= 2) ? Fix<T, m, f, k>(static_cast<T>(T(1) << (m - 2))) : Fix<T, m, f, k>(static_cast<T>(0));
    // 1) xhat_m_share = (x_m_share + r_m_share) in m-ring
    Fix<T, m, f, k> x_hat_m_public = x_hat + bias;

    T msb = (x_hat_m_public.val >> (m - 1)) & T(1);

    // 3) e = 2^m * <r^msb>  (in bw-ring)
    T two_pow_m = (m >= 64) ? T(0) : (T(1) << m);
    T e_share_val = (two_pow_m == T(0)) ? T(0) : (r_msb_share.val * two_pow_m) & bw_mask;

    // 4) t = <e> * (1 - MSB(xhat))
    T one_minus_msb = T(1) - msb;
    T t_share_val = (e_share_val * one_minus_msb) & bw_mask;

    // 5) Return σ·(x̂ − 2^{m−2}) − <r^e> + <t>  (mod 2^bw)
    T bias_val = (m >= 2) ? (T(1) << (m - 2)) : T(0);
    T term_sigma = (party == 1 ? ((x_hat_m_public.val + bw_mask + 1) - bias_val) & bw_mask : T(0));
    T xext_val = (term_sigma + ((t_share_val + bw_mask + 1) - r_e_share.val)) & bw_mask;
    return Fix<T, bw, f, k>(xext_val);
}

// Overload: zero_extend_tensor with provided r_m, r_e, r_msb shares
template <typename T, int bw, int m, int f, int k, int Rank, int Options>
FixTensor<T, bw, f, k, Rank, Options>
zero_extend_tensor(const FixTensor<T, m, f, k, Rank, Options>& x_m_share, ZeroExtendRandomness<T, bw, m, f, k, Rank> randomness, bool reconstructed = false) {
    static_assert(bw > m, "zero_extend_tensor requires bw > m");
    if (mpc_instance == nullptr) throw std::runtime_error("MPC instance not initialized.");
 
    int party = mpc_instance->party;
    const T bw_mask = (bw == 64) ? ~ T(0) : ((T(1) << bw) - 1);

    FixTensor<T, m, f, k, Rank, Options> bias(x_m_share.dimensions());
    
    T bias_val = (m >= 2) ? (T(1) << (m - 2)) : T(0);
    
    bias.setConstant(Fix<T, m, f, k>(bias_val));
    FixTensor<T, m, f, k, Rank, Options>xhat_public = x_m_share;
    if(!reconstructed){
        xhat_public = reconstruct_tensor(x_m_share + randomness.R_M);
    }
    xhat_public = xhat_public + bias;
 
    FixTensor<T, bw, f, k, Rank, Options> result(x_m_share.dimensions());
    T two_pow_m = (m >= 64) ? 0 : (T(1) << m);
 
    for (long long i = 0; i < x_m_share.size(); ++i) {
        T msb = (xhat_public.data()[i].val >> (m - 1)) & T(1);
        T e_share_val = (two_pow_m == T(0)) ? T(0) : (randomness.R_MSB.data()[i].val * two_pow_m) & bw_mask;
        T one_minus_msb = T(1) - msb;
        T t_share_val = (e_share_val * one_minus_msb) & bw_mask;
        
        T term_sigma = (party == 1 ? ((xhat_public.data()[i].val + bw_mask + 1) - bias_val) & bw_mask : T(0));
        T xext_val = (term_sigma + ((t_share_val + bw_mask + 1) - randomness.R_E.data()[i].val)) & bw_mask;
        
        result.data()[i] = Fix<T, bw, f, k>(xext_val);
    }
 
    return result;
}