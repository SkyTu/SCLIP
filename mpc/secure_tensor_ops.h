#pragma once

#include "mpc/fix.h"
#include "mpc/fix_tensor.h"
#include "mpc/tensor_ops.h"
#include "mpc/mpc.h"
#include "mpc/square.h"
#include "utils/random.h"
#include "utils/config.h"
#include "mpc/elementwise_mul.h"
#include "mpc/truncate.h"
#include "math.h"

template <typename T>
size_t get_exp_scalar_random_size(){
    return get_square_scalar_random_size<T>() * RECIPROCAL_NR_ITERS + get_zero_extend_scalar_random_size<T>();
}

template <typename T, int n, int m, int f, int k>
void generate_exp_scalar_randomness(Buffer& p0_buf, Buffer& p1_buf){
    int iters = RECIPROCAL_NR_ITERS;
    for(int i = 0; i < iters; ++i){
        generate_square_scalar_randomness<T, n, m, f, k>(p0_buf, p1_buf);
    }
    generate_zero_extend_scalar_randomness<T, n, m, f, k>(p0_buf, p1_buf);
    return;
}

template <typename T, int n, int m, int f, int k>
Fix<T, n, f, k> exp_scalar(Fix<T, n, f, k> x_n_share, FixTensor<T, m, f, k, 1, Eigen::RowMajor> R, FixTensor<T, n, f, k, 1, Eigen::RowMajor> R_N, 
    FixTensor<T, n, f, k, 1, Eigen::RowMajor> R_SQUARE, FixTensor<T, n, f, k, 1, Eigen::RowMajor> R_MSB, FixTensor<T, n, f, k, 1, Eigen::RowMajor> R_R_MSB, 
    Fix<T, m, f, k> R_M_EXT, Fix<T, n, f, k> R_E_EXT, Fix<T, n, f, k> R_MSB_EXT){
    int iters = RECIPROCAL_NR_ITERS;
    x_n_share = x_n_share * Fix<T, n, f, k>(1.0 / pow(2.0, iters));
    auto init_m_share = truncate_reduce<T, n, f, k>(x_n_share);
    Fix<T, m, f, k> init;
    if (mpc_instance->party==0) {
        init = Fix<T, m, f, k>(1.0) + init_m_share;
    }
    else {
        init = init_m_share;
    }
    auto tmp = reconstruct(init);
    if (mpc_instance->party==0) {
        std::cout << "init.to_float<double>() = " << tmp.template to_float<double>() << std::endl;
    }

    for(int i = 0; i < iters; i++){
        auto tmp = reconstruct(init);
        if (mpc_instance->party==0) {
            std::cout << "tmp.to_float<double>() = " << tmp.template to_float<double>() << std::endl;
        }
        auto res = square_scalar_opt<T, n, m, f, k>(init, R(i), R_N(i), R_SQUARE(i), R_MSB(i), R_R_MSB(i));
        init = truncate_reduce<T, n, f, k>(res);
    }
    Fix<T, n, f, k> res = zero_extend<T, n, m, f, k>(init, R_M_EXT, R_E_EXT, R_MSB_EXT);
    return res;
}

template <typename T, int Rank>
size_t get_exp_random_size(int batch, int row, int col){
    return get_square_random_size<T, Rank>(batch, row, col) * RECIPROCAL_NR_ITERS + get_zero_extend_random_size<T, Rank>(batch, row, col);
}

template <typename T, int n, int m, int f, int k, int Rank>
void generate_exp_randomness(int batch, int row, int col, Buffer& p0_buf, Buffer& p1_buf){
    int iters = RECIPROCAL_NR_ITERS;
    for(int i = 0; i < iters; ++i){
        generate_square_randomness<T, n, m, f, k, Rank>(batch, row, col, p0_buf, p1_buf);
    }
    generate_zero_extend_randomness<T, n, m, f, k, Rank>(batch, row, col, p0_buf, p1_buf);
    return;
}


template <typename T, int n, int m, int f, int k, int Rank, int Options>
FixTensor<T, n, f, k, Rank, Options> exp_tensor(FixTensor<T, n, f, k, Rank, Options> x_n_share, FixTensor<T, m, f, k, Rank, Options> R[], FixTensor<T, n, f, k, Rank, Options> R_N[], 
    FixTensor<T, n, f, k, Rank, Options> R_SQUARE[], FixTensor<T, n, f, k, Rank, Options> R_MSB[], FixTensor<T, n, f, k, Rank, Options> R_R_MSB[], 
    FixTensor<T, m, f, k, Rank, Options> R_M_EXT, FixTensor<T, n, f, k, Rank, Options> R_E_EXT, FixTensor<T, n, f, k, Rank, Options> R_MSB_EXT){
    int iters = RECIPROCAL_NR_ITERS;
    FixTensor<T, n, f, k, Rank, Options> constant(x_n_share.dimensions());
    constant.setConstant(1.0 / pow(2.0, iters));
    x_n_share = x_n_share * constant;
    auto init_m_share = truncate_reduce_tensor(x_n_share);
    FixTensor<T, m, f, k, Rank, Options> init;
    FixTensor<T, m, f, k, Rank, Options> ones(x_n_share.dimensions());
    ones.setConstant(1.0);
    if (mpc_instance->party==0) {
        init = ones + init_m_share;
    }
    else {
        init = init_m_share;
    }
    for(int i = 0; i < iters; i++){
        auto tmp = reconstruct_tensor(init);
        auto res = square_tensor_opt<T, n, m, f, k, Rank, Options>(init, R[i], R_N[i], R_SQUARE[i], R_MSB[i], R_R_MSB[i]);
        init = truncate_reduce_tensor(res);
    }
    auto res = zero_extend_tensor<T, n, m, f, k, Rank, Options>(init, R_M_EXT, R_E_EXT, R_MSB_EXT);
    return res;
}

size_t get_inv_sqrt_random_size(int batch, int row, int col){}

