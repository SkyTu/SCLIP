#ifndef SCLIP_TENSOR_OPS_H
#define SCLIP_TENSOR_OPS_H

#include "fix_tensor.h"
#include <cassert>

// Manual implementation of 2D x 2D tensor contraction (Matrix Multiplication)
// C(m, q) = A(m, n) * B(n, q)
template <typename T, int bw, int f, int k, int Options>
FixTensor<T, bw, f, k, 2, Options> tensor_mul(
    const FixTensor<T, bw, f, k, 2, Options>& a,
    const FixTensor<T, bw, f, k, 2, Options>& b) {

    long long dim_m = a.dimension(0);
    long long dim_n = a.dimension(1);
    long long dim_q = b.dimension(1);
    assert(b.dimension(0) == dim_n);

    FixTensor<T, bw, f, k, 2, Options> result(dim_m, dim_q);
    
    for (long long j_idx = 0; j_idx < dim_q; ++j_idx) {
        for (long long i_idx = 0; i_idx < dim_m; ++i_idx) {
            Fix<T, bw, f, k> sum; // Defaults to 0
            for (long long n_idx = 0; n_idx < dim_n; ++n_idx) {
                sum += a(i_idx, n_idx) * b(n_idx, j_idx);
            }
            result(i_idx, j_idx) = sum;
        }
    }
    return result;
}

// Manual implementation of 3D x 2D tensor contraction (Batch Matrix Multiplication)
// C(batch, m, q) = A(batch, m, n) * B(n, q)
template <typename T, int bw, int f, int k, int Options>
FixTensor<T, bw, f, k, 3, Options> tensor_mul(
    const FixTensor<T, bw, f, k, 3, Options>& a,
    const FixTensor<T, bw, f, k, 2, Options>& b) {

    long long dim_batch = a.dimension(0);
    long long dim_m = a.dimension(1);
    long long dim_n = a.dimension(2);
    long long dim_q = b.dimension(1);
    assert(b.dimension(0) == dim_n);

    FixTensor<T, bw, f, k, 3, Options> result(dim_batch, dim_m, dim_q);

    for(long long batch_idx = 0; batch_idx < dim_batch; ++batch_idx) {
        for (long long j_idx = 0; j_idx < dim_q; ++j_idx) {
            for (long long i_idx = 0; i_idx < dim_m; ++i_idx) {
                Fix<T, bw, f, k> sum; // Defaults to 0
                for (long long n_idx = 0; n_idx < dim_n; ++n_idx) {
                    sum += a(batch_idx, i_idx, n_idx) * b(n_idx, j_idx);
                }
                result(batch_idx, i_idx, j_idx) = sum;
            }
        }
    }
    return result;
}

// Function to change the bit-width format of an entire tensor
template <int new_bw, int new_f, int new_k, typename T, int bw, int f, int k, int Rank, int Options>
FixTensor<T, new_bw, new_f, new_k, Rank, Options>
change_bitwidth(const FixTensor<T, bw, f, k, Rank, Options>& input) {
    FixTensor<T, new_bw, new_f, new_k, Rank, Options> result(input.dimensions());
    
    for (long long i = 0; i < input.size(); ++i) {
        result.data()[i] = input.data()[i].template change_format<new_bw, new_f, new_k>();
    }
    
    return result;
}

// Function to change the bit-width format of an entire tensor
template <int new_bw, int new_f, int new_k, typename T, int bw, int f, int k, int Rank, int Options>
FixTensor<T, new_bw, new_f, new_k, Rank, Options>
extend_locally(const FixTensor<T, bw, f, k, Rank, Options>& input) {
    FixTensor<T, new_bw, new_f, new_k, Rank, Options> result(input.dimensions());
    for (long long i = 0; i < input.size(); ++i) {
        result.data()[i] = input.data()[i];
    }
    return result;
}

template <int new_bw, int new_f, int new_k, typename T, int bw, int f, int k, int Rank, int Options>
FixTensor<T, new_bw, new_f, new_k, Rank, Options>
get_msb(const FixTensor<T, bw, f, k, Rank, Options>& input) {
    FixTensor<T, new_bw, new_f, new_k, Rank, Options> result(input.dimensions());
    for (long long i = 0; i < input.size(); ++i) {
        result.data()[i] = input.data()[i].template get_msb<new_bw, new_f, new_k>();
    }
    return result;
}


#endif //SCLIP_TENSOR_OPS_H
