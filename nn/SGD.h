#ifndef SCLIP_SGD_H
#define SCLIP_SGD_H

#include "mpc/fix_tensor.h"

// Performs an SGD update step on the shares of a weight tensor.
// W_share = W_share - eta * grad_share
// This is a local operation for each party.
template <typename WeightTensorType, typename GradTensorType, typename EtaType>
void sgd_update(WeightTensorType& W_share, const GradTensorType& grad_share, const EtaType& eta) {
    
    // Check if the weight and gradient tensors have compatible dimensions
    static_assert(WeightTensorType::Base::NumIndices == GradTensorType::Base::NumIndices,
                  "Weight and Gradient tensors must have the same rank.");

    // Perform the update: W_share = W_share - (eta * grad_share)
    // We iterate through each element to perform the scalar multiplication and subtraction.
    for (long long i = 0; i < W_share.size(); ++i) {
        W_share.data()[i] -= eta * grad_share.data()[i];
    }
}

#endif // SCLIP_SGD_H
