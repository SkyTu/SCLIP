#pragma once

#include "mpc/fix.h"
#include "mpc/fix_tensor.h"
#include "mpc/tensor_ops.h"
#include "mpc/mpc.h"
#include "utils/random.h"
#include "utils/config.h"

template <typename T, int m, int f, int k, int n, int Rank, int Options>
int get_elementwise_mul_random_size(int batch, int row, int col){
    if(Rank == 3){
        return (batch * row * col + n * row * col + n * row * col + n * row * col + n * row * col + n * row * col + n * row * col + n * row * col) * sizeof(T);
    }else{
        return (row * col + n * row * col + n * row * col + n * row * col + n * row * col + n * row * col + n * row * col + n * row * col) * sizeof(T);
    }
}

template <typename T, int m, int f, int k, int n, int Rank, int Options>
void generate_elementwise_mul_randomness(
    int batch,
    int row,
    int col,
    uint8_t * p0_ptr,
    uint8_t * p1_ptr
){
    FixTensor<T, m, f, k, Rank, Options>& R_X;
    FixTensor<T, n, f, k, Rank, Options>& R_X_N;
    FixTensor<T, n, f, k, Rank, Options>& R_X_MSB;
    FixTensor<T, m, f, k, Rank, Options>& R_Y;
    FixTensor<T, n, f, k, Rank, Options>& R_Y_N;
    FixTensor<T, n, f, k, Rank, Options>& R_Y_MSB;
    FixTensor<T, n, f, k, Rank, Options>& R_XY;
    FixTensor<T, n, f, k, Rank, Options>& R_X_RYMSB;
    FixTensor<T, n, f, k, Rank, Options>& R_XMSB_Y;
    if(Rank == 3){
        R_X = FixTensor<T, m, f, k, 3>(batch, row, col);
        R_X_N = FixTensor<T, n, f, k, 3>(batch, row, col);
        R_X_MSB = FixTensor<T, n, f, k, 3>(batch, row, col);
        R_Y = FixTensor<T, m, f, k, 3>(batch, row, col);
        R_Y_N = FixTensor<T, n, f, k, 3>(batch, row, col);
        R_Y_MSB = FixTensor<T, n, f, k, 3>(batch, row, col);
        R_XY = FixTensor<T, n, f, k, 3>(batch, row, col);
    }else{
        R_X = FixTensor<T, m, f, k, Rank>(row, col);
        R_X_N = FixTensor<T, n, f, k, Rank>(row, col);
        R_X_MSB = FixTensor<T, n, f, k, Rank>(row, col);
        R_Y = FixTensor<T, m, f, k, Rank>(row, col);
        R_Y_N = FixTensor<T, n, f, k, Rank>(row, col);
        R_Y_MSB = FixTensor<T, n, f, k, Rank>(row, col);
        R_XY = FixTensor<T, n, f, k, Rank>(row, col);
        R_X_RYMSB = FixTensor<T, n, f, k, Rank>(row, col);
        R_XMSB_Y = FixTensor<T, n, f, k, Rank>(row, col);
    }
    Random rg;
    for(long long i = 0; i < R_X.size(); ++i) {
        T val = rg.template randomGE<T>(1, m)[0];
        R_X.data()[i] = Fix<T, m, f, k>(val);
        R_X_N.data()[i] = Fix<T, n, f, k>(val);
        R_X_MSB.data()[i] = R_X.data()[i].template get_msb<n, f, k>();
        val = rg.template randomGE<T>(1, m)[0];
        R_Y.data()[i] = Fix<T, m, f, k>(val);
        R_Y_N.data()[i] = Fix<T, n, f, k>(val);
        R_Y_MSB.data()[i] = R_Y.data()[i].template get_msb<n, f, k>();
    }
    R_XY = (R_X_N * R_Y_N);
    R_X_RYMSB = R_X_N * R_Y_MSB;
    R_XMSB_Y = R_X_MSB * R_Y_N;
    secret_share_and_write_tensor(R_X, p0_ptr, p1_ptr);
    secret_share_and_write_tensor(R_Y, p0_ptr, p1_ptr);
    secret_share_and_write_tensor(R_X_N, p0_ptr, p1_ptr);
    secret_share_and_write_tensor(R_Y_N, p0_ptr, p1_ptr);
    secret_share_and_write_tensor(R_X_MSB, p0_ptr, p1_ptr);
    secret_share_and_write_tensor(R_Y_MSB, p0_ptr, p1_ptr);
    secret_share_and_write_tensor(R_XY, p0_ptr, p1_ptr);
    secret_share_and_write_tensor(R_X_RYMSB, p0_ptr, p1_ptr);
    secret_share_and_write_tensor(R_XMSB_Y, p0_ptr, p1_ptr);
    return;
}

// Optimized Element-wise Multiplication Protocol from PDF
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
        term1.setConstant(Fix<T,n,f,k>(0));
    }
    else{
        term1 = x_hat_prime_n * y_hat_prime_n;
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