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


FixTensor<T, n, f, k, Rank, Options> inv_sqrt_tensor(FixTensor<T, n, f, k, Rank, Options> x_n_share, FixTensor<T, m, f, k, Rank, Options> R[], FixTensor<T, n, f, k, Rank, Options> R_N[], 
    FixTensor<T, n, f, k, Rank, Options> R_SQUARE[], FixTensor<T, n, f, k, Rank, Options> R_MSB[], FixTensor<T, n, f, k, Rank, Options> R_R_MSB[], 
    FixTensor<T, m, f, k, Rank, Options> R_M_EXT[], FixTensor<T, n, f, k, Rank, Options> R_E_EXT[], FixTensor<T, n, f, k, Rank, Options> R_MSB_EXT[]){
    int iters = INV_SQRT_NR_ITERS;
    FixTensor<T, n, f, k, Rank, Options> constant_n_tmp(x_n_share.dimensions());
    FixTensor<T, m, f, k, Rank, Options> constant_m_tmp(x_n_share.dimensions());
    // init = (e^{- (x/2 + 0.2)} * 2.2 + 0.2) - x / 1024
    constant_n_tmp.setConstant(0.5);
    x_n_share = x_n_share * constant_n_tmp;
    auto tmp_m_share = truncate_reduce_tensor(x_n_share);
    FixTensor<T, m, f, k, Rank, Options> init_m_share;
    constant_n_tmp.setConstant(0.2);
    init_m_share =  - (constant_n_tmp + tmp_m_share);
    init_n_share = zero_extend_tensor<T, n, m, f, k, Rank, Options>(init_m_share, R_M_EXT[0], R_E_EXT[0], R_MSB_EXT[0]);
    init_n_share = exp_tensor<T, n, m, f, k, Rank, Options>(init_n_share, R, R_N, R_SQUARE, R_MSB, R_R_MSB, R_M_EXT[1], R_E_EXT[1], R_MSB_EXT[1]);
    
    constant_n_tmp.setConstant(2.2);
    init_n_share = constant_n_tmp * init_n_share;

    init_m_share = truncate_reduce_tensor(init_n_share);
    constant_m_tmp.setConstant(0.2);
    init_m_share = constant_m_tmp + init_m_share;
    
    constant_n_tmp.setConstant(1.0 / 1024.0);
    init_m_share = init_m_share + truncate_reduce_tensor(constant_n_tmp * x_n_share);

    FixTensor<T, m, f, k, Rank, Options> x_m_share = change_bitwidth<m, f, k, T, n, f, k, Rank, Options>(x_n_share);
    FixTensor<T, n, f, k, Rank, Options> x_mul_init_n_share;
    FixTensor<T, m, f, k, Rank, Options> x_mul_init_m_share;
    FixTensor<T, n, f, k, Rank, Options> term1;
    for(int i = 0; i < iters; i++){
        reconstruct_tensor_parallel(init_m_share + R[i], x_m_share + R_X_M_SHARE[2*i])
        // y ** 2
        auto sq = square_tensor_opt<T, n, m, f, k, Rank, Options>(init_m_share, R[i], R_N[i], R_SQUARE[i], R_MSB[i], R_R_MSB[i]);
        auto sq_m_share = truncate_reduce_tensor(sq);
        // x * y
        x_mul_init_n_share = elementwise_mul_opt<T, n, m, f, k, Rank, Options>(x_m_share, init_m_share, R_X_M_SHARE_1[i], R_X_N_SHARE_1[i], R_X_MSB_N_SHARE_1[i], R[i], R_N[i], R_MSB[i], R_RXY_N_SHARE_1[i], R_RX_MSBY_N_SHARE_1[i], R_RXY_MSB_N_SHARE_1[i]);
        x_mul_init_m_share = truncate_reduce_tensor(x_mul_init_n_share);
        // x * y * y ** 2
        term1 = elementwise_mul_opt<T, n, m, f, k, Rank, Options>(x_mul_init_n_share, sq_m_share, R_X_M_SHARE_2[i], R_X_N_SHARE_2[i], R_X_MSB_N_SHARE_2[i], R_Y_M_SHARE_2[i], R_Y_N_SHARE_2[i], R_Y_MSB_N_SHARE_2[i], R_RXY_N_SHARE_2[i], R_RX_MSBY_N_SHARE_2[i], R_RXY_MSB_N_SHARE_2[i]);
        auto tmp = truncate_reduce_tensor(term1);
    }
    
    init_m_share = truncate_reduce_tensor(init_n_share);
    init_n_share = zero_extend_tensor<T, n, m, f, k, Rank, Options>(init_m_share, R_M_EXT[3], R_E_EXT[3], R_MSB_EXT[3]);
    return init_n_share;
}