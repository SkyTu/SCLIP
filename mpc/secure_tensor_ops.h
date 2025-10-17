#pragma once

#include "mpc/fix.h"
#include "mpc/fix_tensor.h"
#include "mpc/tensor_ops.h"
#include "mpc/mpc.h"
#include "mpc/square.h"
#include "utils/random.h"
#include "utils/config.h"
#include "mpc/elementwise_mul.h"

template <typename T, int m, int f, int k, int n>
void generate_exp_randomness(Buffer& p0_buf, Buffer& p1_buf){
    int iters = RECIPROCAL_NR_ITERS;
    for(int i = 0; i < iters; ++i){
        generate_square_randomness<T, m, f, k, Rank, Options>(batch, row, col, p0_buf, p1_buf);
    }
    return;
}


void exp(){
    int iters = RECIPROCAL_NR_ITERS;
    for(int i = 0; i < iters; ++i){
        
    }
}

void generate_exp_tensor_randomness(){

}

void exp_tensor(){

}

void sqrt(){

}