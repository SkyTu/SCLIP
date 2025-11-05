#pragma once

#include "mpc/fix.h"
#include "mpc/fix_tensor.h"
#include "mpc/tensor_ops.h"
#include "mpc/mpc.h"
#include "utils/random.h"
#include "utils/config.h"

template <typename T, int n, int m, int f, int k, int Rank>
struct SquareRandomness{
    FixTensor<T, m, f, k, Rank> R;
    FixTensor<T, n, f, k, Rank> R_N;
    FixTensor<T, n, f, k, Rank> R_SQUARE;
    FixTensor<T, n, f, k, Rank> R_MSB;
    FixTensor<T, n, f, k, Rank> R_R_MSB;
};

template <typename T, int n, int m, int f, int k, int Rank>
SquareRandomness<T, n, m, f, k, Rank> read_square_randomness(MPC& mpc, int batch, int row, int col){
    SquareRandomness<T, n, m, f, k, Rank> randomness;
    if constexpr(Rank == 3){
        randomness.R.resize(batch, row, col);
        randomness.R_N.resize(batch, row, col);
        randomness.R_SQUARE.resize(batch, row, col);
        randomness.R_MSB.resize(batch, row, col);
        randomness.R_R_MSB.resize(batch, row, col);
    }
    else{
        randomness.R.resize(row, col);
        randomness.R_N.resize(row, col);
        randomness.R_SQUARE.resize(row, col);
        randomness.R_MSB.resize(row, col);
        randomness.R_R_MSB.resize(row, col);
    }
    mpc.read_fixtensor_share(randomness.R);
    mpc.read_fixtensor_share(randomness.R_N);
    mpc.read_fixtensor_share(randomness.R_SQUARE);
    mpc.read_fixtensor_share(randomness.R_MSB);
    mpc.read_fixtensor_share(randomness.R_R_MSB);
    return randomness;
}

template <typename T, int Rank>
int get_square_random_size(int batch, int row, int col){
    size_t m_size = 0;
    size_t n_size = 0;
    if(Rank == 3){
        m_size = batch * row * col * sizeof(T);
        n_size = batch * row * col * sizeof(T);
    } else {
        m_size = row * col * sizeof(T);
        n_size = row * col * sizeof(T);
    }
    return m_size + 4 * n_size;
}

template <typename T>
size_t get_square_scalar_random_size(){
    return 5 * sizeof(T);
}

template <typename T, int n, int m, int f, int k>
void generate_square_scalar_randomness(Buffer& p0_buf, Buffer& p1_buf){
    Fix<T, m, f, k> R;
    Fix<T, n, f, k> R_N;
    Fix<T, n, f, k> R_SQUARE;
    Fix<T, n, f, k> R_MSB;
    Fix<T, n, f, k> R_R_MSB;
    Random rg;
    T r_val = rg.template randomGE<T>(1, m)[0];
    // T r_val = 1;
    R = Fix<T, m, f, k>(r_val);
    R_N = Fix<T, n, f, k>(r_val);
    R_SQUARE = R_N * R_N;
    R_MSB = R.template get_msb<n, f, k>();
    R_R_MSB = R_N * R_MSB;
    secret_share_and_write_scalar<Fix<T, m, f, k>>(R, p0_buf, p1_buf);
    secret_share_and_write_scalar<Fix<T, n, f, k>>(R_N, p0_buf, p1_buf);
    secret_share_and_write_scalar<Fix<T, n, f, k>>(R_SQUARE, p0_buf, p1_buf);
    secret_share_and_write_scalar<Fix<T, n, f, k>>(R_MSB, p0_buf, p1_buf);
    secret_share_and_write_scalar<Fix<T, n, f, k>>(R_R_MSB, p0_buf, p1_buf);
}


template <typename T, int n, int m, int f, int k,  int Rank>
void generate_square_randomness(Buffer& p0_buf, Buffer& p1_buf, int batch, int row, int col){
    FixTensor<T, m, f, k, Rank> R;
    FixTensor<T, n, f, k, Rank> R_N;
    FixTensor<T, n, f, k, Rank> R_SQUARE;
    FixTensor<T, n, f, k, Rank> R_MSB;
    FixTensor<T, n, f, k, Rank> R_R_MSB;
    if constexpr(Rank == 3){
        R.resize(batch, row, col);
        R_N.resize(batch, row, col);
        R_SQUARE.resize(batch, row, col);
        R_MSB.resize(batch, row, col);
        R_R_MSB.resize(batch, row, col);
        Random rg;
        T* val = rg.template randomGE<T>(batch * row * col, m);
        for (int i = 0; i < batch; i++) {
            for (int j = 0; j < row; j++) {
                for (int z = 0; z < col; z++) {
                    R(i, j, z) = Fix<T, m, f, k>(val[i * row * col + j * col + z]);
                }
            }
        }
        delete[] val; // <--- 必须加上这一行！
        R_N = extend_locally<n, f, k>(R);
        R_SQUARE = R_N * R_N;
        R_MSB = get_msb<n, f, k>(R);
        R_R_MSB = R_N * R_MSB;
        secret_share_and_write_tensor(R, p0_buf, p1_buf);
        secret_share_and_write_tensor(R_N, p0_buf, p1_buf);
        secret_share_and_write_tensor(R_SQUARE, p0_buf, p1_buf);
        secret_share_and_write_tensor(R_MSB, p0_buf, p1_buf);
        secret_share_and_write_tensor(R_R_MSB, p0_buf, p1_buf);
    } else {
        R.resize(row, col);
        R_N.resize(row, col);
        R_SQUARE.resize(row, col);
        R_MSB.resize(row, col);
        R_R_MSB.resize(row, col);
        Random rg;
        T* val = rg.template randomGE<T>(row * col, m);
        for (int i = 0; i < row; i++) {
            for (int j = 0; j < col; j++) {
                R(i, j) = Fix<T, m, f, k>(val[i * col + j]);
            }
        }
        delete[] val; // <--- 必须加上这一行！
        R_N = extend_locally<n, f, k>(R);
        R_SQUARE = R_N * R_N;
        R_MSB = get_msb<n, f, k>(R);
        R_R_MSB = R_N * R_MSB;
        secret_share_and_write_tensor(R, p0_buf, p1_buf);
        secret_share_and_write_tensor(R_N, p0_buf, p1_buf);
        secret_share_and_write_tensor(R_SQUARE, p0_buf, p1_buf);
        secret_share_and_write_tensor(R_MSB, p0_buf, p1_buf);
        secret_share_and_write_tensor(R_R_MSB, p0_buf, p1_buf);
    }
}

template <typename T, int n, int m, int f, int k>
Fix<T, n, f, k> square_scalar_opt(Fix<T, m, f, k>x_m_share, Fix<T, m, f, k> R, Fix<T, n, f, k> R_N, Fix<T, n, f, k> R_SQUARE, Fix<T, n, f, k> R_MSB, Fix<T, n, f, k> R_R_MSB){
    auto x_hat = reconstruct(x_m_share + R);

    T two_pow_m_minus_2_val = (m < 2 || m - 2 >= 64) ? 0 : (T(1) << (m - 2));
    Fix<T, m, f, k> const_term_m = Fix<T, m, f, k>(two_pow_m_minus_2_val);
    Fix<T, m, f, k> x_hat_prime_m = x_hat + const_term_m;

    T two_pow_m_val = (m >= 64) ? 0 : (T(1) << m);

    Fix<T, n, f, k> t_x = (Fix<T, n, f, k>(1) - x_hat_prime_m.template get_msb<n, f, k>()) * Fix<T, n, f, k>(two_pow_m_val);

    Fix<T, n, f, k> x_hat_prime_n = Fix<T, n, f, k>(x_hat_prime_m.val);
    Fix<T, n, f, k> const_term_n = Fix<T, n, f, k>(two_pow_m_minus_2_val);
    
    x_hat_prime_n = x_hat_prime_n - const_term_n;
    
    Fix<T, n, f, k> term1;
    if (mpc_instance->party == 0){
        term1 = x_hat_prime_n * x_hat_prime_n;
    }
    else{
        term1 = Fix<T,n,f,k>(0);
    }
    Fix<T, n, f, k> term2 = x_hat_prime_n * R_N;
    Fix<T, n, f, k> term3 = x_hat_prime_n * t_x * R_MSB;
    Fix<T, n, f, k> term4 = t_x * R_R_MSB;
    return term1 + R_SQUARE - term2 - term2 + term3 + term3 - term4 - term4;

}

//extend & square
template <typename T, int n, int m, int f, int k, int Rank, int Options>
FixTensor<T, n, f, k, Rank, Options> square_tensor_opt(FixTensor<T, m, f, k, Rank, Options> x_m_share, SquareRandomness<T, n, m, f, k, Rank> randomness, bool reconstructed = false)
{
    auto x_hat = reconstructed ? x_m_share : reconstruct_tensor(x_m_share + randomness.R);

    T two_pow_m_minus_2_val = (m < 2 || m - 2 >= 64) ? 0 : (T(1) << (m - 2));
    FixTensor<T, m, f, k, Rank, Options> const_term_m(x_hat.dimensions());
    const_term_m.setConstant(Fix<T,m,f,k>(two_pow_m_minus_2_val));
    FixTensor<T, m, f, k, Rank, Options> x_hat_prime = x_hat + const_term_m;

    T two_pow_m_val = (m >= 64) ? 0 : (T(1) << m);
    auto ones_n = FixTensor<T, n, f, k, Rank, Options>(x_hat.dimensions());
    ones_n.setConstant(Fix<T,n,f,k>(1));
    auto two_pow_m_n = FixTensor<T, n, f, k, Rank, Options>(x_hat.dimensions());
    two_pow_m_n.setConstant(Fix<T,n,f,k>(two_pow_m_val));

    auto t_x = (ones_n - get_msb<n,f,k>(x_hat_prime)) * two_pow_m_n;

    FixTensor<T, n, f, k, Rank, Options> x_hat_prime_n = extend_locally<n,f,k>(x_hat_prime);
    FixTensor<T, n, f, k, Rank, Options> const_term_n(x_hat.dimensions());
    const_term_n.setConstant(Fix<T,n,f,k>(two_pow_m_minus_2_val));

    x_hat_prime_n = x_hat_prime_n - const_term_n;

    auto term1 = FixTensor<T, n, f, k, Rank, Options>(x_hat.dimensions());
    if (mpc_instance->party == 0){
        term1 = x_hat_prime_n * x_hat_prime_n;
    }
    else{
        term1.setConstant(Fix<T,n,f,k>(0));
    }
    auto term2 = x_hat_prime_n * randomness.R_N;
    auto term3 = x_hat_prime_n * t_x * randomness.R_MSB;
    auto term4 = t_x * randomness.R_R_MSB;
    return term1 + randomness.R_SQUARE - term2 - term2 + term3 + term3 - term4 - term4;
}
