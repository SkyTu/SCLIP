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
        result.data()[i].val = input.data()[i].val;
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

// Sum-reduce a 3D tensor to a 2D tensor along a specified axis
template <int axis, typename T, int bw, int f, int k, int Options>
FixTensor<T, bw, f, k, 2, Options>
sum_reduce_tensor(const FixTensor<T, bw, f, k, 3, Options>& input) {
    long long dim0 = input.dimension(0);
    long long dim1 = input.dimension(1);
    long long dim2 = input.dimension(2);

    FixTensor<T, bw, f, k, 2, Options> result;
    if (axis == 0) {
        result.resize(dim1, dim2);
    } else if (axis == 1) {
        result.resize(dim0, dim2);
    } else if (axis == 2) {
        result.resize(dim0, dim1);
    }
    result.setZero();
    
    for (long long i = 0; i < dim0; ++i) {
        for (long long j = 0; j < dim1; ++j) {
            for (long long l = 0; l < dim2; ++l) {
                if constexpr (axis == 0) {
                    result(j, l) += input(i, j, l);
                } else if constexpr (axis == 1) {
                    result(i, l) += input(i, j, l);
                } else if constexpr (axis == 2) {
                    result(i, j) += input(i, j, l);
                }
            }
        }
    }
    return result;
}

// Sum-reduce a 2D tensor to a 1D tensor by summing each row (axis 1)
template <int axis, typename T, int bw, int f, int k, int Options>
FixTensor<T, bw, f, k, 1, Options>
sum_reduce_tensor(const FixTensor<T, bw, f, k, 2, Options>& input) {
    long long dim_rows = input.dimension(0);
    long long dim_cols = input.dimension(1);
    FixTensor<T, bw, f, k, 1, Options> result;
    if constexpr (axis == 0) {
        result.resize(dim_cols);
    } else if constexpr (axis == 1) {
        result.resize(dim_rows);
    }
    result.setZero();

    for (long long j = 0; j < dim_cols; ++j) {
        for (long long i = 0; i < dim_rows; ++i) {
            if constexpr (axis == 0) {
                result(j) += input(i, j);
            } else if constexpr (axis == 1) {
                result(i) += input(i, j);
            }
        }
    }
    return result;
}


#endif //SCLIP_TENSOR_OPS_H
