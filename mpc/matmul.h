#pragma once

#include "mpc/fix.h"
#include "mpc/fix_tensor.h"
#include "mpc/tensor_ops.h"
#include "mpc/mpc.h"
#include "utils/random.h"

template <typename T, int bw, int f, int k, int RankU, int RankV, int RankZ>
struct MatmulRandomness{
    FixTensor<T, bw, f, k, RankU> U;
    FixTensor<T, bw, f, k, RankV> V;
    FixTensor<T, bw, f, k, RankZ> Z;
};

template <typename T, int bw, int f, int k, int RankU, int RankV, int RankZ>
MatmulRandomness<T, bw, f, k, RankU, RankV, RankZ> read_matmul_randomness(MPC& mpc, int batch, int m, int n, int q){
    MatmulRandomness<T, bw, f, k, RankU, RankV, RankZ> randomness;
    assert(RankU == 3 || RankU == 2);
    assert(RankV == 3 || RankV == 2);
    assert(RankZ == 3 || RankZ == 2);
    if constexpr (RankU == 3){
        randomness.U.resize(batch, m, n);
    }
    else{
        randomness.U.resize(m, n);
    }
    if constexpr (RankV == 3){
        randomness.V.resize(batch, n, q);
    }
    else{
        randomness.V.resize(n, q);
    }
    if constexpr (RankZ == 3){
        randomness.Z.resize(batch, m, q);
    }
    else{
        randomness.Z.resize(m, q);
    }
    mpc.read_fixtensor_share(randomness.U);
    mpc.read_fixtensor_share(randomness.V);
    mpc.read_fixtensor_share(randomness.Z);
    return randomness;
}

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
void generate_matmul_randomness(Buffer& p0_buf, Buffer& p1_buf, int B, int m, int n, int q){
    FixTensor<T, bw, f, k, RankU> U;
    FixTensor<T, bw, f, k, RankV> V;
    FixTensor<T, bw, f, k, RankZ> Z;
    if constexpr (RankU == 3) {
        assert(B != -1);
        U.resize(B, m, n);
        V.resize(n, q);
        Z.resize(B, m, q);
    } else {
        U.resize(m, n);
        V.resize(n, q);
        Z.resize(m, q);
    }
    U.setRandom();
    V.setRandom();
    Z = tensor_mul(U, V);
    secret_share_and_write_tensor(U, p0_buf, p1_buf);
    secret_share_and_write_tensor(V, p0_buf, p1_buf);
    secret_share_and_write_tensor(Z, p0_buf, p1_buf);
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
    const MatmulRandomness<T, bw, f, k, RankX, RankY, RankX>& randomness,
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
    FixTensorX e;
    FixTensorY f_rec_val;

    if (e_ptr == nullptr && f_rec_ptr == nullptr) {
        e = x_share - randomness.U;
        f_rec_val = y_share - randomness.V;
        reconstruct_tensor_parallel(e, f_rec_val);
    }
    else if (e_ptr == nullptr && f_rec_ptr != nullptr) {
        e = x_share - randomness.U;
        f_rec_val = reconstruct_tensor(*f_rec_ptr - randomness.V);
    }
    else if (e_ptr != nullptr && f_rec_ptr == nullptr) {
        e = reconstruct_tensor(*e_ptr - randomness.U);
        f_rec_val = y_share - randomness.V;
    }
    else{
        e = *e_ptr;
        f_rec_val = *f_rec_ptr;
    }
    

    FixTensorZ term1 = tensor_mul(e, y_share);
    FixTensorZ term2 = tensor_mul(x_share, f_rec_val);
    FixTensorZ output_share = term1 + term2 + randomness.Z;
    if (party == 1) {
        FixTensorZ ef_prod = tensor_mul(e, f_rec_val);
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
    const MatmulRandomness<T, bw, f, k, 3, 2, 3>& randomness,
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
    FixTensorX e;
    FixTensorY f_rec_val;
    if (e_ptr == nullptr && f_rec_ptr == nullptr) {
        e = x_share - randomness.U;
        f_rec_val = y_share - randomness.V;
        reconstruct_tensor_parallel(e, f_rec_val);
    }
    else if (e_ptr == nullptr && f_rec_ptr != nullptr) {
        e = x_share - randomness.U;
        f_rec_val = reconstruct_tensor(*f_rec_ptr - randomness.V);
    }
    else if (e_ptr != nullptr && f_rec_ptr == nullptr) {
        e = reconstruct_tensor(*e_ptr - randomness.U);
        f_rec_val = y_share - randomness.V;
    }
    else{
        e = *e_ptr;
        f_rec_val = *f_rec_ptr;
    }
    FixTensorZ term1 = tensor_mul(e, y_share);
    FixTensorZ term2 = tensor_mul(x_share, f_rec_val);
    FixTensorZ output_share = term1 + term2 + randomness.Z;
    if (party == 1) {
        FixTensorZ ef_prod = tensor_mul(e, f_rec_val);
        output_share = output_share - ef_prod;
    }
    return output_share;
}
