#pragma once

#include "mpc/fix.h"
#include "mpc/fix_tensor.h"
#include "mpc/tensor_ops.h"
#include "mpc/mpc.h"
#include "utils/random.h"
#include "utils/config.h"

template <typename T, int n, int m, int f, int k, int Rank, int Options = Eigen::RowMajor>
struct ElementwiseMulRandomness{
    FixTensor<T, m, f, k, Rank, Options> r_x_m;
    FixTensor<T, m, f, k, Rank, Options> r_y_m;
    FixTensor<T, n, f, k, Rank, Options> r_x_n;
    FixTensor<T, n, f, k, Rank, Options> r_y_n;
    FixTensor<T, n, f, k, Rank, Options> r_x_msb;
    FixTensor<T, n, f, k, Rank, Options> r_y_msb;
    FixTensor<T, n, f, k, Rank, Options> r_xy;
    FixTensor<T, n, f, k, Rank, Options> r_x_rymsb;
    FixTensor<T, n, f, k, Rank, Options> r_xmsb_y;
};

template <typename T, int n, int m, int f, int k, int Rank, int Options = Eigen::RowMajor>
ElementwiseMulRandomness<T, n, m, f, k, Rank, Options> read_elementwise_mul_randomness(MPC& mpc, int batch, int row, int col){
    ElementwiseMulRandomness<T, n, m, f, k, Rank, Options> randomness;
    assert(Rank == 3 || Rank == 2);
    if constexpr (Rank == 3){ // Use if constexpr
        randomness.r_x_m.resize(batch, row, col);
        randomness.r_y_m.resize(batch, row, col);
        randomness.r_x_n.resize(batch, row, col);
        randomness.r_y_n.resize(batch, row, col);
        randomness.r_x_msb.resize(batch, row, col);
        randomness.r_y_msb.resize(batch, row, col);
        randomness.r_xy.resize(batch, row, col);
        randomness.r_x_rymsb.resize(batch, row, col);
        randomness.r_xmsb_y.resize(batch, row, col);
    }
    else{ // This branch is only compiled if Rank is not 3
        randomness.r_x_m.resize(row, col);
        randomness.r_y_m.resize(row, col);
        randomness.r_x_n.resize(row, col);
        randomness.r_y_n.resize(row, col);
        randomness.r_x_msb.resize(row, col);
        randomness.r_y_msb.resize(row, col);
        randomness.r_xy.resize(row, col);
        randomness.r_x_rymsb.resize(row, col);
        randomness.r_xmsb_y.resize(row, col);
    }
    mpc.read_fixtensor_share(randomness.r_x_m);
    mpc.read_fixtensor_share(randomness.r_y_m);    
    mpc.read_fixtensor_share(randomness.r_x_n);    
    mpc.read_fixtensor_share(randomness.r_y_n);    
    mpc.read_fixtensor_share(randomness.r_x_msb);    
    mpc.read_fixtensor_share(randomness.r_y_msb);    
    mpc.read_fixtensor_share(randomness.r_xy);    
    mpc.read_fixtensor_share(randomness.r_x_rymsb);    
    mpc.read_fixtensor_share(randomness.r_xmsb_y);    
    return randomness;
}

template <typename T, int Rank>
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

template <typename T, int n, int m, int f, int k, int Rank, int Options = Eigen::RowMajor>
void generate_elementwise_mul_randomness(
    Buffer& p0_buf,
    Buffer& p1_buf,
    int batch,
    int row,
    int col
){
    if constexpr (Rank == 3) {
        FixTensor<T, m, f, k, Rank, Options> r_x_m(batch, row, col), r_y_m(batch, row, col);
        FixTensor<T, n, f, k, Rank, Options> r_x_n(batch, row, col), r_y_n(batch, row, col), r_x_msb(batch, row, col), r_y_msb(batch, row, col);
        FixTensor<T, n, f, k, Rank, Options> r_xy(batch, row, col), r_x_rymsb(batch, row, col), r_xmsb_y(batch, row, col);
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
template <typename T, int n, int m, int f, int k, int Rank, int Options = Eigen::RowMajor>
auto elementwise_mul_opt(
    const FixTensor<T, m, f, k, Rank, Options>& x_m_share,
    const FixTensor<T, m, f, k, Rank, Options>& y_m_share,
    const ElementwiseMulRandomness<T, n, m, f, k, Rank, Options>& randomness,
    bool x_reconstructed = false,
    bool y_reconstructed = false
) -> FixTensor<T, n, f, k, Rank, Options>
{
    if (mpc_instance == nullptr) throw std::runtime_error("MPC instance not initialized.");
    
    auto x_hat = x_m_share;
    auto y_hat = y_m_share;
    if (!x_reconstructed && !y_reconstructed){
        x_hat = x_m_share + randomness.r_x_m;
        y_hat = y_m_share + randomness.r_y_m;
        reconstruct_tensor_parallel(x_hat, y_hat);
    }
    else if (x_reconstructed && !y_reconstructed){
        y_hat = reconstruct_tensor(y_m_share + randomness.r_y_m);
    }
    else if (!x_reconstructed && y_reconstructed){
        x_hat = reconstruct_tensor(x_m_share + randomness.r_x_m);
    }
    

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
    auto term2 = x_hat_prime_n * randomness.r_y_n;
    auto term3 = x_hat_prime_n * t_y * randomness.r_y_msb;
    auto term4 = randomness.r_x_n * y_hat_prime_n;
    auto term5 = randomness.r_xy;
    auto term6 = t_y * randomness.r_x_rymsb;
    auto term7 = t_x * randomness.r_x_msb * y_hat_prime_n;
    auto term8 = t_x * randomness.r_xmsb_y;
    // auto term9 = t_x * t_y * rx_msby_msb_n_share;
    
    FixTensor<T, n, f, k, Rank, Options> result = term1 - term2 + term3 - term4 + term5 - term6 + term7 - term8; // + term9;
    return result;
}