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

template <typename T, int n, int m, int f, int k, int Rank>
size_t get_inv_sqrt_random_size(int batch, int row, int col){
    size_t size = 0, m_size = 0, n_size = 0;
    size += get_zero_extend_random_size<T, Rank>(batch, row, col);
    size += get_exp_random_size<T, Rank>(batch, row, col);
    for(int i = 0; i < INV_SQRT_NR_ITERS; ++i){
        size += get_square_random_size<T, Rank>(batch, row, col);
        if(Rank == 3){
            m_size = batch * row * col * sizeof(T);
            n_size = batch * row * col * sizeof(T);
        } else {
            m_size = row * col * sizeof(T);
            n_size = row * col * sizeof(T);
        }
        size += m_size + 5 * n_size;
        size += get_elementwise_mul_random_size<T, Rank>(batch, row, col);
        size += get_zero_extend_random_size<T, Rank>(batch, row, col);
    }
    return size;
}

// by default, the inv_sqrt input will be 3D
template <typename T, int n, int m, int f, int k, int Rank>
void generate_inv_sqrt_randomness(int batch, int row, int col, Buffer& p0_buf, Buffer& p1_buf){
    int iters = INV_SQRT_NR_ITERS;
    generate_zero_extend_randomness<T, n, m, f, k, Rank>(batch, row, col, p0_buf, p1_buf);
    generate_exp_randomness<T, n, m, f, k, Rank>(batch, row, col, p0_buf, p1_buf);
    for(int i = 0; i < iters; ++i){
        if constexpr(Rank == 3){
            FixTensor<T, m, f, k, Rank> R(batch, row, col);
            FixTensor<T, n, f, k, Rank> R_N(batch, row, col);
            FixTensor<T, n, f, k, Rank> R_SQUARE(batch, row, col);
            FixTensor<T, n, f, k, Rank> R_MSB(batch, row, col);
            FixTensor<T, n, f, k, Rank> R_R_MSB(batch, row, col);
            R.setRandom();
            // R.setConstant(Fix<T,m,f,k>(0));
            R_N = extend_locally<n, f, k>(R);
            R_SQUARE = R_N * R_N;
            R_MSB = get_msb<n, f, k>(R_N);
            R_R_MSB = R_N * R_MSB;
            secret_share_and_write_tensor(R, p0_buf, p1_buf);
            secret_share_and_write_tensor(R_N, p0_buf, p1_buf);
            secret_share_and_write_tensor(R_SQUARE, p0_buf, p1_buf);
            secret_share_and_write_tensor(R_MSB, p0_buf, p1_buf);
            secret_share_and_write_tensor(R_R_MSB, p0_buf, p1_buf);

            FixTensor<T, m, f, k, Rank, Eigen::RowMajor> r_x_m(batch, row, col), r_y_m(batch, row, col);
            FixTensor<T, n, f, k, Rank, Eigen::RowMajor> r_x_n(batch, row, col), r_y_n(batch, row, col), r_x_msb(batch, row, col), r_y_msb(batch, row, col);
            FixTensor<T, n, f, k, Rank, Eigen::RowMajor> r_xy(batch, row, col), r_x_rymsb(batch, row, col), r_xmsb_y(batch, row, col);
            r_x_m.setRandom();
            r_y_m = R;
            r_x_n = extend_locally<n,f,k>(r_x_m);
            r_y_n = R_N;
            r_x_msb = get_msb<n,f,k>(r_x_n);
            r_y_msb = R_MSB;
            r_xy = r_x_n * r_y_n;
            r_x_rymsb = r_x_n * r_y_msb;
            r_xmsb_y = r_x_msb * r_y_n;
            secret_share_and_write_tensor(r_x_m, p0_buf, p1_buf);
            secret_share_and_write_tensor(r_x_n, p0_buf, p1_buf);
            secret_share_and_write_tensor(r_x_msb, p0_buf, p1_buf);
            secret_share_and_write_tensor(r_xy, p0_buf, p1_buf);
            secret_share_and_write_tensor(r_x_rymsb, p0_buf, p1_buf);
            secret_share_and_write_tensor(r_xmsb_y, p0_buf, p1_buf);
        } else {
            FixTensor<T, m, f, k, Rank> R(row, col);
            FixTensor<T, n, f, k, Rank> R_N(row, col);
            FixTensor<T, n, f, k, Rank> R_SQUARE(row, col);
            FixTensor<T, n, f, k, Rank> R_MSB(row, col);
            FixTensor<T, n, f, k, Rank> R_R_MSB(row, col);
            R.setRandom();
            // R.setConstant(Fix<T,m,f,k>(0));
            R_N = extend_locally<n, f, k>(R);
            R_SQUARE = R_N * R_N;
            R_MSB = get_msb<n, f, k>(R_N);
            R_R_MSB = R_N * R_MSB;
            secret_share_and_write_tensor(R, p0_buf, p1_buf);
            secret_share_and_write_tensor(R_N, p0_buf, p1_buf);
            secret_share_and_write_tensor(R_SQUARE, p0_buf, p1_buf);
            secret_share_and_write_tensor(R_MSB, p0_buf, p1_buf);
            secret_share_and_write_tensor(R_R_MSB, p0_buf, p1_buf);
            FixTensor<T, m, f, k, Rank, Eigen::RowMajor> r_x_m(row, col), r_y_m(row, col);
            FixTensor<T, n, f, k, Rank, Eigen::RowMajor> r_x_n(row, col), r_y_n(row, col), r_x_msb(row, col), r_y_msb(row, col);
            FixTensor<T, n, f, k, Rank, Eigen::RowMajor> r_xy(row, col), r_x_rymsb(row, col), r_xmsb_y(row, col);
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
            secret_share_and_write_tensor(r_x_n, p0_buf, p1_buf);
            secret_share_and_write_tensor(r_x_msb, p0_buf, p1_buf);
            secret_share_and_write_tensor(r_xy, p0_buf, p1_buf);
            secret_share_and_write_tensor(r_x_rymsb, p0_buf, p1_buf);
            secret_share_and_write_tensor(r_xmsb_y, p0_buf, p1_buf);
        }
        generate_elementwise_mul_randomness<T, n, m, f, k, Rank, Eigen::RowMajor>(batch, row, col, p0_buf, p1_buf);
        generate_zero_extend_randomness<T, n, m, f, k, Rank>(batch, row, col, p0_buf, p1_buf);
    }
}

template <typename T, int n, int m, int f, int k, int Rank>
FixTensor<T, n, f, k, Rank> inv_sqrt_tensor(FixTensor<T, n, f, k, Rank> x_n_share, FixTensor<T, m, f, k, Rank> R[], FixTensor<T, n, f, k, Rank> R_N[], 
    FixTensor<T, n, f, k, Rank> R_SQUARE[], FixTensor<T, n, f, k, Rank> R_MSB[], FixTensor<T, n, f, k, Rank> R_R_MSB[], 
    FixTensor<T, m, f, k, Rank> R_M_EXT[], FixTensor<T, n, f, k, Rank> R_E_EXT[], FixTensor<T, n, f, k, Rank> R_MSB_EXT[], 
    FixTensor<T, m, f, k, Rank> R_X_M_SHARE_1[], FixTensor<T, n, f, k, Rank> R_X_N_SHARE_1[], FixTensor<T, n, f, k, Rank> R_X_MSB_N_SHARE_1[], 
    FixTensor<T, n, f, k, Rank> R_RXY_N_SHARE_1[], FixTensor<T, n, f, k, Rank> R_RX_MSBY_N_SHARE_1[], FixTensor<T, n, f, k, Rank> R_RXY_MSB_N_SHARE_1[], 
    FixTensor<T, m, f, k, Rank> R_X_M_SHARE_2[], FixTensor<T, n, f, k, Rank> R_X_N_SHARE_2[], FixTensor<T, n, f, k, Rank> R_X_MSB_N_SHARE_2[], 
    FixTensor<T, n, f, k, Rank> R_Y_M_SHARE_2[], FixTensor<T, n, f, k, Rank> R_Y_N_SHARE_2[], FixTensor<T, n, f, k, Rank> R_Y_MSB_N_SHARE_2[], 
    FixTensor<T, n, f, k, Rank> R_RXY_N_SHARE_2[], FixTensor<T, n, f, k, Rank> R_RX_MSBY_N_SHARE_2[], FixTensor<T, n, f, k, Rank> R_RXY_MSB_N_SHARE_2[]){
    int iters = INV_SQRT_NR_ITERS;
    FixTensor<T, n, f, k, Rank> constant_n_tmp(x_n_share.dimensions());
    FixTensor<T, m, f, k, Rank> constant_m_tmp(x_n_share.dimensions());
    FixTensor<T, n, f, k, Rank> init_n_share(x_n_share.dimensions());
    FixTensor<T, m, f, k, Rank> init_m_share(x_n_share.dimensions());
    // init = (e^{- (x/2 + 0.2)} * 2.2 + 0.2) - x / 1024
    constant_n_tmp.setConstant(0.5);
    x_n_share = x_n_share * constant_n_tmp;
    auto tmp_m_share = truncate_reduce_tensor(x_n_share);
    constant_n_tmp.setConstant(0.2);
    init_m_share =  - (constant_n_tmp + tmp_m_share);
    init_n_share = zero_extend_tensor<T, n, m, f, k, Rank>(init_m_share, R_M_EXT[0], R_E_EXT[0], R_MSB_EXT[0]);
    init_n_share = exp_tensor<T, n, m, f, k, Rank>(init_n_share, R, R_N, R_SQUARE, R_MSB, R_R_MSB, R_M_EXT[1], R_E_EXT[1], R_MSB_EXT[1]);
    
    constant_n_tmp.setConstant(2.2);
    init_n_share = constant_n_tmp * init_n_share;

    init_m_share = truncate_reduce_tensor(init_n_share);
    constant_m_tmp.setConstant(0.2);
    init_m_share = constant_m_tmp + init_m_share;
    
    constant_n_tmp.setConstant(1.0 / 1024.0);
    init_m_share = init_m_share + truncate_reduce_tensor(constant_n_tmp * x_n_share);

    FixTensor<T, m, f, k, Rank> x_m_share = change_bitwidth<m, f, k, T, n, f, k, Rank>(x_n_share);
    FixTensor<T, n, f, k, Rank> x_mul_init_n_share;
    FixTensor<T, m, f, k, Rank> x_mul_init_m_share;
    FixTensor<T, n, f, k, Rank> term1;
    for(int i = 0; i < iters; i++){
        reconstruct_tensor_parallel(init_m_share + R[i], x_m_share + R_X_M_SHARE_1[i]);
        // y ** 2
        auto sq = square_tensor_opt<T, n, m, f, k, Rank>(init_m_share, R[i], R_N[i], R_SQUARE[i], R_MSB[i], R_R_MSB[i]);
        auto sq_m_share = truncate_reduce_tensor(sq);
        // x * y
        x_mul_init_n_share = elementwise_mul_opt<T, n, m, f, k, Rank>(x_m_share, init_m_share, R_X_M_SHARE_1[i], R_X_N_SHARE_1[i], R_X_MSB_N_SHARE_1[i], R[i], R_N[i], R_MSB[i], R_RXY_N_SHARE_1[i], R_RX_MSBY_N_SHARE_1[i], R_RXY_MSB_N_SHARE_1[i]);
        x_mul_init_m_share = truncate_reduce_tensor(x_mul_init_n_share);
        // x * y * y ** 2
        term1 = elementwise_mul_opt<T, n, m, f, k, Rank>(x_mul_init_n_share, sq_m_share, R_X_M_SHARE_2[i], R_X_N_SHARE_2[i], R_X_MSB_N_SHARE_2[i], R_Y_M_SHARE_2[i], R_Y_N_SHARE_2[i], R_Y_MSB_N_SHARE_2[i], R_RXY_N_SHARE_2[i], R_RX_MSBY_N_SHARE_2[i], R_RXY_MSB_N_SHARE_2[i]);
        auto tmp = truncate_reduce_tensor(term1);
        constant_m_tmp.setConstant(Fix<T, m, f, k>(3));
        auto term2 = constant_m_tmp * init_m_share;
        init_m_share = term2 - term1;
        
        init_n_share = zero_extend_tensor<T, n, m, f, k, Rank>(init_m_share, R_M_EXT[i+1], R_E_EXT[i+1], R_MSB_EXT[i+1]);
        constant_n_tmp.setConstant(0.5);
        init_m_share = truncate_reduce_tensor(constant_n_tmp * init_n_share);
    }
    return init_m_share;
}
