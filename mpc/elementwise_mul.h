#pragma once

#include "mpc/fix.h"
#include "mpc/fix_tensor.h"
#include "mpc/tensor_ops.h"
#include "mpc/mpc.h"
#include "utils/random.h"
#include "utils/config.h"

template <typename T, int m, int f, int k, int n, int Rank, int Options>
int get_elementwise_mul_random_size(int batch, int row, int col){
    size_t m_size = 0;
    size_t n_size = 0;
    if(Rank == 3){
        m_size = batch * row * col * sizeof(T);
        n_size = batch * row * col * sizeof(T);
    } else {
        m_size = row * col * sizeof(T);
        n_size = row * col * sizeof(T);
    }
    return 2 * m_size + 7 * n_size;
}

template <typename T, int m, int f, int k, int n, int Rank, int Options>
void generate_elementwise_mul_randomness(
    int batch,
    int row,
    int col,
    Buffer& p0_buf,
    Buffer& p1_buf
){
    if constexpr (Rank == 3) {
        FixTensor<T, m, f, k, Rank, Options> r_x_m(batch, row, col), r_y_m(batch, row, col);
        FixTensor<T, n, f, k, Rank, Options> r_x_n(batch, row, col), r_y_n(batch, row, col), r_x_msb(batch, row, col), r_y_msb(batch, row, col);
        FixTensor<T, n, f, k, Rank, Options> r_xy(batch, row, col), r_x_rymsb(batch, row, col), r_xmsb_y(batch, row, col);
        r_x_m.setRadnom();
        r_y_m.setRadnom();
        r_x_n = extend_locally<n,f,k>(r_x_m);
        r_y_n = extend_locally<n,f,k>(r_y_m);
        r_x_msb = get_msb<n,f,k>(r_x_n);
        r_y_msb = get_msb<n,f,k>(r_y_n);
        r_xy = r_x_n * r_y_n;
        r_x_rymsb = r_x_n * r_y_msb;
        r_xmsb_y = r_x_msb * r_y_n;
        secret_share_and_write_tensor(r_x_m, p0_buf, p1_buf);
        secret_share_and_write_tensor(r_y_m, p0_buf, p1_buf);
        secret_share_and_write_tensor(r_x_n, p0_buf, p1_buf);
        secret_share_and_write_tensor(r_y_n, p0_buf, p1_buf);
        secret_share_and_write_tensor(r_x_msb, p0_buf, p1_buf);
        secret_share_and_write_tensor(r_y_msb, p0_buf, p1_buf);
        secret_share_and_write_tensor(r_xy, p0_buf, p1_buf);
        secret_share_and_write_tensor(r_x_rymsb, p0_buf, p1_buf);
        secret_share_and_write_tensor(r_xmsb_y, p0_buf, p1_buf);
    } else {
        FixTensor<T, m, f, k, Rank, Options> r_x_m(row, col), r_y_m(row, col);
        FixTensor<T, n, f, k, Rank, Options> r_x_n(row, col), r_y_n(row, col), r_x_msb(row, col), r_y_msb(row, col);
        FixTensor<T, n, f, k, Rank, Options> r_xy(row, col), r_x_rymsb(row, col), r_xmsb_y(row, col);
        r_x_m.setRandom();
        r_y_m.setRandom();
        r_x_n = extend_locally<n,f,k>(r_x_m);
        r_y_n = extend_locally<n,f,k>(r_y_m);
        r_x_msb = get_msb<n,f,k>(r_x_n);
        r_y_msb = get_msb<n,f,k>(r_y_n);
        r_xy = r_x_n * r_y_n;
        r_x_rymsb = r_x_n * r_y_msb;
        r_xmsb_y = r_x_msb * r_y_n;
        secret_share_and_write_tensor(r_x_m, p0_buf, p1_buf);
        secret_share_and_write_tensor(r_y_m, p0_buf, p1_buf);
        secret_share_and_write_tensor(r_x_n, p0_buf, p1_buf);
        secret_share_and_write_tensor(r_y_n, p0_buf, p1_buf);
        secret_share_and_write_tensor(r_x_msb, p0_buf, p1_buf);
        secret_share_and_write_tensor(r_y_msb, p0_buf, p1_buf);
        secret_share_and_write_tensor(r_xy, p0_buf, p1_buf);
        secret_share_and_write_tensor(r_x_rymsb, p0_buf, p1_buf);
        secret_share_and_write_tensor(r_xmsb_y, p0_buf, p1_buf);
    }
}

// Optimized Element-wise Multiplication Protocol (Elementwise Mul & Extend)
template <typename T, int m, int f, int k, int n, int Rank, int Options>
auto elementwise_mul_opt(
    const FixTensor<T, m, f, k, Rank, Options>& x_m_share,
    const FixTensor<T, m, f, k, Rank, Options>& y_m_share,
    const FixTensor<T, m, f, k, Rank, Options>& rx_m_share,
    const FixTensor<T, n, f, k, Rank, Options>& rx_n_share,
    const FixTensor<T, n, f, k, Rank, Options>& rx_msb_n_share,
    const FixTensor<T, m, f, k, Rank, Options>& ry_m_share,
    const FixTensor<T, n, f, k, Rank, Options>& ry_n_share,
    const FixTensor<T, n, f, k, Rank, Options>& ry_msb_n_share,
    const FixTensor<T, n, f, k, Rank, Options>& rxy_n_share,
    const FixTensor<T, n, f, k, Rank, Options>& rx_msby_n_share,
    const FixTensor<T, n, f, k, Rank, Options>& rxy_msb_n_share
    // const FixTensor<T, n, f, k, Rank, Options>& rx_msby_msb_n_share
) -> FixTensor<T, n, f, k, Rank, Options>
{
    if (mpc_instance == nullptr) throw std::runtime_error("MPC instance not initialized.");
    
    auto x_hat = reconstruct_tensor(x_m_share + rx_m_share);
    auto y_hat = reconstruct_tensor(y_m_share + ry_m_share);

    T two_pow_m_minus_2_val = (m < 2 || m - 2 >= 64) ? 0 : (T(1) << (m - 2));
    FixTensor<T, m, f, k, Rank, Options> const_term_m(x_hat.dimensions());
    const_term_m.setConstant(Fix<T,m,f,k>(two_pow_m_minus_2_val));
    FixTensor<T, m, f, k, Rank, Options> x_hat_prime = x_hat + const_term_m;
    FixTensor<T, m, f, k, Rank, Options> y_hat_prime = y_hat + const_term_m;
    

    T two_pow_m_val = (m >= 64) ? 0 : (T(1) << m);
    auto ones_n = FixTensor<T, n, f, k, Rank, Options>(x_hat.dimensions());
    ones_n.setConstant(Fix<T,n,f,k>(1));
    auto two_pow_m_n = FixTensor<T, n, f, k, Rank, Options>(x_hat.dimensions());
    two_pow_m_n.setConstant(Fix<T,n,f,k>(two_pow_m_val));
    
    auto t_x = (ones_n - get_msb<n,f,k>(x_hat_prime)) * two_pow_m_n;
    auto t_y = (ones_n - get_msb<n,f,k>(y_hat_prime)) * two_pow_m_n;

    FixTensor<T, n, f, k, Rank, Options> x_hat_prime_n = extend_locally<n,f,k>(x_hat_prime);
    FixTensor<T, n, f, k, Rank, Options> y_hat_prime_n = extend_locally<n,f,k>(y_hat_prime);
    FixTensor<T, n, f, k, Rank, Options> const_term_n(x_hat.dimensions());
    const_term_n.setConstant(Fix<T,n,f,k>(two_pow_m_minus_2_val));
    x_hat_prime_n = x_hat_prime_n - const_term_n;
    y_hat_prime_n = y_hat_prime_n - const_term_n;
    auto term1 = FixTensor<T, n, f, k, Rank, Options>(x_hat.dimensions());
    if (mpc_instance->party == 0){
        term1 = x_hat_prime_n * y_hat_prime_n;
    }
    else{
        term1.setConstant(Fix<T,n,f,k>(0));
    }
    auto term2 = x_hat_prime_n * ry_n_share;
    auto term3 = x_hat_prime_n * t_y * ry_msb_n_share;
    auto term4 = rx_n_share * y_hat_prime_n;
    auto term5 = rxy_n_share;
    auto term6 = t_y * rxy_msb_n_share;
    auto term7 = t_x * rx_msb_n_share * y_hat_prime_n;
    auto term8 = t_x * rx_msby_n_share;
    // auto term9 = t_x * t_y * rx_msby_msb_n_share;
    
    FixTensor<T, n, f, k, Rank, Options> result = term1 - term2 + term3 - term4 + term5 - term6 + term7 - term8; // + term9;
    return result;
}