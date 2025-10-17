#pragma once

#include "mpc/fix.h"
#include "mpc/fix_tensor.h"
#include "mpc/tensor_ops.h"
#include "mpc/mpc.h"
#include "utils/random.h"

template <typename T, int bw, int f, int k, int RankU, int RankV, int RankZ>
int get_matmul_random_size(int m, int n, int q, int B = -1){
    if (RankU == 3){
        assert(B != -1);
        return (B * m * n + n * q + B * m * q) * sizeof(T);
    }
    else{
        return (m * n + n * q + m * q) * sizeof(T);
    }
}

template <typename T, int bw, int f, int k, int RankU, int RankV, int RankZ>
void generate_matmul_randomness(Buffer& p0_buf, Buffer& p1_buf, int m, int n, int q, int B = -1){
    if constexpr (RankU == 3) {
        assert(B != -1);
        FixTensor<T, bw, f, k, RankU> U(B, m, n);
        FixTensor<T, bw, f, k, RankV> V(n, q);
        FixTensor<T, bw, f, k, RankZ> Z(B, m, q);
        U.setRandom();
        V.setRandom();
        Z = tensor_mul(U, V);
        secret_share_and_write_tensor(U, p0_buf, p1_buf);
        secret_share_and_write_tensor(V, p0_buf, p1_buf);
        secret_share_and_write_tensor(Z, p0_buf, p1_buf);
    } else {
        FixTensor<T, bw, f, k, RankU> U(m, n);
        FixTensor<T, bw, f, k, RankV> V(n, q);
        FixTensor<T, bw, f, k, RankZ> Z(m, q);
        U.setRandom();
        V.setRandom();
        Z = tensor_mul(U, V);
        secret_share_and_write_tensor(U, p0_buf, p1_buf);
        secret_share_and_write_tensor(V, p0_buf, p1_buf);
        secret_share_and_write_tensor(Z, p0_buf, p1_buf);
    }
}

// Overload: secure_matmul with provided tensor Beaver triple shares (U,V,Z)
template<
    typename T, int bw, int f, int k,
    int RankX, int RankY,
    template<typename, int, int, int, int, int> class FixTensorT
>
auto secure_matmul(
    const FixTensorT<T, bw, f, k, RankX, Eigen::RowMajor>& x_share,
    const FixTensorT<T, bw, f, k, RankY, Eigen::RowMajor>& y_share,
    const FixTensorT<T, bw, f, k, RankX, Eigen::RowMajor>& u_share,
    const FixTensorT<T, bw, f, k, RankY, Eigen::RowMajor>& v_share,
    const FixTensorT<T, bw, f, k, (RankX - 1 + RankY - 1), Eigen::RowMajor>& z_share,
    const FixTensorT<T, bw, f, k, RankX, Eigen::RowMajor>* e_ptr = nullptr,
    const FixTensorT<T, bw, f, k, RankY, Eigen::RowMajor>* f_rec_ptr = nullptr
) -> FixTensorT<T, bw, f, k, RankX - 1 + RankY - 1, Eigen::RowMajor>
{
    if (mpc_instance == nullptr) throw std::runtime_error("MPC instance not initialized.");
    if (mpc_instance->M != 2) throw std::runtime_error("This secure_matmul function only supports 2 parties.");

    int party = mpc_instance->party;

    using FixTensorX = FixTensorT<T, bw, f, k, RankX, Eigen::RowMajor>;
    using FixTensorY = FixTensorT<T, bw, f, k, RankY, Eigen::RowMajor>;
    using FixTensorZ = FixTensorT<T, bw, f, k, (RankX - 1 + RankY - 1), Eigen::RowMajor>;

    FixTensorX e_val;
    if (e_ptr == nullptr) {
        FixTensorX e_share = x_share - u_share;
        e_val = reconstruct_tensor(e_share);
    }
    const FixTensorX& e = (e_ptr == nullptr) ? e_val : *e_ptr;

    FixTensorY f_rec_val;
    if (f_rec_ptr == nullptr){
        FixTensorY f_share = y_share - v_share;
        f_rec_val = reconstruct_tensor(f_share);
    }
    const FixTensorY& f_rec = (f_rec_ptr == nullptr) ? f_rec_val : *f_rec_ptr;

    FixTensorZ term1 = tensor_mul(e, y_share);
    FixTensorZ term2 = tensor_mul(x_share, f_rec);
    FixTensorZ output_share = term1 + term2 + z_share;
    if (party == 1) {
        FixTensorZ ef_prod = tensor_mul(e, f_rec);
        output_share = output_share - ef_prod;
    }
    return output_share;
}


// Overload: 3Dx2D with provided triples U,V,Z
template<
    typename T, int bw, int f, int k,
    template<typename, int, int, int, int, int> class FixTensorT
>
auto secure_matmul(
    const FixTensorT<T, bw, f, k, 3, Eigen::RowMajor>& x_share,
    const FixTensorT<T, bw, f, k, 2, Eigen::RowMajor>& y_share,
    const FixTensorT<T, bw, f, k, 3, Eigen::RowMajor>& u_share,
    const FixTensorT<T, bw, f, k, 2, Eigen::RowMajor>& v_share,
    const FixTensorT<T, bw, f, k, 3, Eigen::RowMajor>& z_share,
    const FixTensorT<T, bw, f, k, 3, Eigen::RowMajor>* e_ptr = nullptr,
    const FixTensorT<T, bw, f, k, 2, Eigen::RowMajor>* f_rec_ptr = nullptr
) -> FixTensorT<T, bw, f, k, 3, Eigen::RowMajor>
{
    if (mpc_instance == nullptr) throw std::runtime_error("MPC instance not initialized.");
    if (mpc_instance->M != 2) throw std::runtime_error("This secure_matmul function only supports 2 parties.");

    int party = mpc_instance->party;

    using FixTensorX = FixTensorT<T, bw, f, k, 3, Eigen::RowMajor>;
    using FixTensorY = FixTensorT<T, bw, f, k, 2, Eigen::RowMajor>;
    using FixTensorZ = FixTensorT<T, bw, f, k, 3, Eigen::RowMajor>;

    FixTensorX e_val;
    if (e_ptr == nullptr) {
        FixTensorX e_share = x_share - u_share;
        e_val = reconstruct_tensor(e_share);
    }
    const FixTensorX& e = (e_ptr == nullptr) ? e_val : *e_ptr;

    FixTensorY f_rec_val;
    if (f_rec_ptr == nullptr){
        FixTensorY f_share = y_share - v_share;
        f_rec_val = reconstruct_tensor(f_share);
    }
    const FixTensorY& f_rec = (f_rec_ptr == nullptr) ? f_rec_val : *f_rec_ptr;

    FixTensorZ term1 = tensor_mul(e, y_share);
    FixTensorZ term2 = tensor_mul(x_share, f_rec);
    FixTensorZ output_share = term1 + term2 + z_share;
    if (party == 1) {
        FixTensorZ ef_prod = tensor_mul(e, f_rec);
        output_share = output_share - ef_prod;
    }
    return output_share;
}
