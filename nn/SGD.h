#pragma once

#include "mpc/fix_tensor.h"
#include "mpc/tensor_ops.h"
#include "mpc/mpc.h"

// SGD update rule: W = W - lr * dW
// This function performs the update on secret shared tensors.
template <
    typename T, int W_BW, int F, int K_INT, int RANK, int OPTIONS,
    typename GradTensorType, typename EtaType,
    typename RmTensorType, typename ReTensorType, typename RmsbTensorType
>
void sgd_update(
    FixTensor<T, W_BW, F, K_INT, RANK, OPTIONS>& W_share,
    const GradTensorType& grad_share,
    const EtaType& eta,
    const RmTensorType& r_m_share,
    const ReTensorType& r_e_share,
    const RmsbTensorType& r_msb_share
) {
    // 1. Calculate the update term: grad * lr
    auto update_term_wide_expr = grad_share * eta;
    // The result of Eigen expression needs to be evaluated into a concrete tensor
    FixTensor<T, W_BW, F, K_INT, RANK, OPTIONS> update_term_wide = update_term_wide_expr;

    // 2. Truncate the wide product to reduce fractional bits
    // The result is in a smaller bitwidth ring (m-bit)
    auto update_term_trunc = truncate_reduce_tensor(update_term_wide);

    // 3. Securely extend the truncated value back to the original bitwidth (bw-bit)
    auto update_term_extended = zero_extend_tensor(
        update_term_trunc, r_m_share, r_e_share, r_msb_share
    );

    // 4. Perform the final update
    W_share = W_share - update_term_extended;
}
