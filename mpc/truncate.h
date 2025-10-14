#pragma once

#include "mpc/fix.h"
#include "mpc/fix_tensor.h"
#include "mpc/tensor_ops.h"
#include "mpc/mpc.h"
#include "utils/random.h"

template <typename T, int smallBW, int BW, int F, int K>
void generate_zero_extend_randomness(FixTensor<T, smallBW, F, K, 2>& r_m_plain, FixTensor<T, BW, F, K, 2>& r_e_plain, FixTensor<T, BW, F, K, 2>& r_msb_plain){
    // For SGD zero_extend
    Random random_gen;
    auto val = random_gen.template randomGE<T>(r_m_plain.size(), smallBW);
    for(int i = 0; i < r_m_plain.size(); ++i) {
        r_m_plain.data()[i] = Fix<T, smallBW, F, K>(val[i]);
        r_e_plain.data()[i] = r_m_plain.data()[i];
        // std::cout << "r_e_val: " << r_e.data()[i].val << std::endl;
        r_msb_plain.data()[i] = r_m_plain.data()[i].template get_msb<BW, F, K>();
    }
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

template <typename T, int m, int f, int k, int bw>
Fix<T, bw, f, k> zero_extend(const Fix<T, m, f, k>& x_m_share, const Fix<T, m, f, k>& r_m_share, const Fix<T, bw, f, k>& r_e_share, const Fix<T, bw, f, k>& r_msb_share) {
    static_assert(bw > m, "zero_extend requires bw > m");
    if (mpc_instance == nullptr) throw std::runtime_error("MPC instance not initialized.");
    
    const T bw_mask = (bw == 64) ? ~T(0) : ((T(1) << bw) - 1);

    int M = mpc_instance->M;
    int party = mpc_instance->party;
    Fix<T, m, f, k> bias = (party == 1 && m >= 2) ? Fix<T, m, f, k>(static_cast<T>(T(1) << (m - 2))) : Fix<T, m, f, k>(static_cast<T>(0));
    // 1) xhat_m_share = (x_m_share + r_m_share) in m-ring
    Fix<T, m, f, k> xhat_share_m = x_m_share + r_m_share + bias;

    // 2) Reconstruct xhat (public), then MSB at bit (m-1)
    Fix<T, m, f, k> xhat_m_public = reconstruct(xhat_share_m);
    T msb = (xhat_m_public.val >> (m - 1)) & T(1);

    // 3) e = 2^m * <r^msb>  (in bw-ring)
    T two_pow_m = (m >= 64) ? T(0) : (T(1) << m);
    T e_share_val = (two_pow_m == T(0)) ? T(0) : (r_msb_share.val * two_pow_m) & bw_mask;

    // 4) t = <e> * (1 - MSB(xhat))
    T one_minus_msb = T(1) - msb;
    T t_share_val = (e_share_val * one_minus_msb) & bw_mask;

    // 5) Return σ·(x̂ − 2^{m−2}) − <r^e> + <t>  (mod 2^bw)
    T bias_val = (m >= 2) ? (T(1) << (m - 2)) : T(0);
    T term_sigma = (party == 1 ? ((xhat_m_public.val + bw_mask + 1) - bias_val) & bw_mask : T(0));
    T xext_val = (term_sigma + ((t_share_val + bw_mask + 1) - r_e_share.val)) & bw_mask;
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
template <typename T, int m, int f, int k, int Rank, int Options, int bw>
FixTensor<T, bw, f, k, Rank, Options>
zero_extend_tensor(const FixTensor<T, m, f, k, Rank, Options>& x_m_share,
                   const FixTensor<T, m, f, k, Rank, Options>& r_m_share,
                   const FixTensor<T, bw, f, k, Rank, Options>& r_e_share,
                   const FixTensor<T, bw, f, k, Rank, Options>& r_msb_share) {
    static_assert(bw > m, "zero_extend_tensor requires bw > m");
    if (mpc_instance == nullptr) throw std::runtime_error("MPC instance not initialized.");
 
    int party = mpc_instance->party;
    const T bw_mask = (bw == 64) ? ~ T(0) : ((T(1) << bw) - 1);

    FixTensor<T, m, f, k, Rank, Options> bias(x_m_share.dimensions());
    
    T bias_val = (m >= 2) ? (T(1) << (m - 2)) : T(0);
    
    bias.setConstant(Fix<T, m, f, k>(bias_val));
    auto xhat_m_share = x_m_share + r_m_share;
    FixTensor<T, m, f, k, Rank, Options>xhat_public = reconstruct_tensor(xhat_m_share) + bias;
 
    FixTensor<T, bw, f, k, Rank, Options> result(x_m_share.dimensions());
    T two_pow_m = (m >= 64) ? 0 : (T(1) << m);
 
    for (long long i = 0; i < x_m_share.size(); ++i) {
        T msb = (xhat_public.data()[i].val >> (m - 1)) & T(1);
        T e_share_val = (two_pow_m == T(0)) ? T(0) : (r_msb_share.data()[i].val * two_pow_m) & bw_mask;
        T one_minus_msb = T(1) - msb;
        T t_share_val = (e_share_val * one_minus_msb) & bw_mask;
        
        T term_sigma = (party == 1 ? ((xhat_public.data()[i].val + bw_mask + 1) - bias_val) & bw_mask : T(0));
        T xext_val = (term_sigma + ((t_share_val + bw_mask + 1) - r_e_share.data()[i].val)) & bw_mask;
        
        result.data()[i] = Fix<T, bw, f, k>(xext_val);
    }
 
    return result;
}