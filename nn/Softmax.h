#pragma once

#include "mpc/mpc.h"
#include "mpc/fix_tensor.h"
#include "mpc/tensor_ops.h"
#include "mpc/truncate.h"
#include "mpc/secure_tensor_ops.h"
#include "mpc/elementwise_mul.h"
#include "mpc/matmul.h"
#include "utils/random.h"

struct SoftmaxLayerParams {
    int B;       // Batch size
};


template <typename T, int BW, int smallBW, int F, int K_INT>
struct SoftmaxRandomness {
    ExpScalarRandomness<T, BW, smallBW, F, K_INT> exp_scalar_randomness;
    ElementwiseMulRandomness<T, BW, smallBW, F, K_INT, 2> elementwise_mul_randomness_temperature;
    ZeroExtendRandomness<T, BW, smallBW, F, K_INT, 2> zero_extend_randomness_temperature;
    ExpRandomness<T, BW, smallBW, F, K_INT, 2> exp_randomness;
    ReciprocalRandomness<T, BW, smallBW, F, K_INT, 1> reciprocal_randomness;
    ElementwiseMulRandomness<T, BW, smallBW, F, K_INT, 2> elementwise_mul_randomness;
    ZeroExtendRandomness<T, BW, smallBW, F, K_INT, 2> zero_extend_randomness;
    ZeroExtendRandomness<T, BW, smallBW, F, K_INT, 2> zero_extend_randomness_loss;
    ElementwiseMulRandomness<T, BW, smallBW, F, K_INT, 2> elementwise_mul_randomness_update;
    ElementwiseMulRandomness<T, BW, smallBW, F, K_INT, 1> elementwise_mul_randomness_update_lambda;
    ZeroExtendRandomness<T, BW, smallBW, F, K_INT, 1> zero_extend_randomness_update_dlambda[2];
};

template <typename T, int BW, int smallBW, int F, int K_INT>
class SoftmaxLayer {
    public:
        double init_lambda = 1.3;
        Fix<T, BW, F, K_INT> lambda{init_lambda};
        Fix<T, smallBW, F, K_INT> temperature;
        SoftmaxLayerParams p;
        SoftmaxRandomness<T, BW, smallBW, F, K_INT> randomness;
        FixTensor<T, smallBW, F, K_INT, 2> loss_m;
        FixTensor<T, smallBW, F, K_INT, 2> input_rec;
        SoftmaxLayer(const SoftmaxLayerParams& params) : p(params) {}
        ~SoftmaxLayer() {}

        size_t getRandomnessSize() {
            size_t total_size = 0;
            size_t tmp_size = 0;
            total_size += get_exp_scalar_random_size<T>(false);
            total_size += get_elementwise_mul_random_size<T, 2>(-1, p.B, p.B);
            total_size += get_zero_extend_random_size<T, 2>(-1, p.B, p.B);
            total_size += get_exp_random_size<T, 2>(-1, p.B, p.B);
            total_size += get_reciprocal_non_extend_random_size<T, 1>(-1, -1, 2 * p.B);
            total_size += get_elementwise_mul_random_size<T, 2>(-1, 2 * p.B, p.B);
            total_size += get_zero_extend_random_size<T, 2>(-1, 2 * p.B, p.B);
            total_size += get_zero_extend_random_size<T, 2>(-1, p.B, p.B);
            total_size += get_elementwise_mul_random_size<T, 2>(-1, p.B, p.B);
            total_size += get_elementwise_mul_random_size<T, 1>(-1, 1, 1);
            total_size += get_zero_extend_random_size<T, 1>(-1, 1, 1) * 2;
            return total_size;
        }

        SoftmaxRandomness<T, BW, smallBW, F, K_INT> read_softmax_randomness(MPC& mpc){
            SoftmaxRandomness<T, BW, smallBW, F, K_INT> randomness;
            randomness.exp_scalar_randomness = read_exp_scalar_randomness<T, BW, smallBW, F, K_INT>(mpc, false);
            randomness.elementwise_mul_randomness_temperature = read_elementwise_mul_randomness<T, BW, smallBW, F, K_INT, 2>(mpc, -1, p.B, p.B);
            randomness.zero_extend_randomness_temperature = read_zero_extend_randomness<T, BW, smallBW, F, K_INT, 2>(mpc, -1, p.B, p.B);
            randomness.exp_randomness = read_exp_randomness<T, BW, smallBW, F, K_INT, 2>(mpc, -1, p.B, p.B);
            randomness.reciprocal_randomness = read_reciprocal_non_extend_randomness<T, BW, smallBW, F, K_INT, 1>(mpc, -1, -1, 2 * p.B);
            randomness.elementwise_mul_randomness = read_elementwise_mul_randomness<T, BW, smallBW, F, K_INT, 2>(mpc, -1, 2 * p.B, p.B);
            randomness.zero_extend_randomness = read_zero_extend_randomness<T, BW, smallBW, F, K_INT, 2>(mpc, -1, 2 * p.B, p.B);
            randomness.zero_extend_randomness_loss = read_zero_extend_randomness<T, BW, smallBW, F, K_INT, 2>(mpc, -1, p.B, p.B);
            randomness.elementwise_mul_randomness_update = read_elementwise_mul_randomness<T, BW, smallBW, F, K_INT, 2>(mpc, -1, p.B, p.B);
            randomness.elementwise_mul_randomness_update_lambda = read_elementwise_mul_randomness<T, BW, smallBW, F, K_INT, 1>(mpc, -1, 1, 1);
            randomness.zero_extend_randomness_update_dlambda[0] = read_zero_extend_randomness<T, BW, smallBW, F, K_INT, 1>(mpc, -1, 1, 1);
            randomness.zero_extend_randomness_update_dlambda[1] = read_zero_extend_randomness<T, BW, smallBW, F, K_INT, 1>(mpc, -1, 1, 1);
            return randomness;
        }

        void generate_softmax_randomness(Buffer& p0_buf, Buffer& p1_buf){
            uint8_t * pre_ptr = p0_buf.ptr;
            generate_exp_scalar_randomness<T, BW, smallBW, F, K_INT>(p0_buf, p1_buf, false);
            FixTensor<T, smallBW, F, K_INT, 2> r_x_m_tensor(p.B, p.B);
            FixTensor<T, smallBW, F, K_INT, 2> sample_tensor(p.B, p.B);
            Random rg;
            
            T* r_x_m = rg.template randomGE<T>(p.B * p.B, smallBW);
            T* sample_val = rg.template randomGE<T>(p.B, smallBW);
            for(int i = 0; i < p.B; i++){
                for(int j = 0; j < p.B; j++){
                    r_x_m_tensor(i, j) = Fix<T, smallBW, F, K_INT>(r_x_m[i * p.B + j]);
                    sample_tensor(i, j) = Fix<T, smallBW, F, K_INT>(sample_val[i]);
                }
            }
            std::cout << "r_x_m_tensor: " << r_x_m_tensor << std::endl;
            generate_elementwise_mul_randomness<T, BW, smallBW, F, K_INT, 2, Eigen::RowMajor>(p0_buf, p1_buf, -1, p.B, p.B, &r_x_m_tensor, &sample_tensor);
            generate_zero_extend_randomness<T, BW, smallBW, F, K_INT, 2>(p0_buf, p1_buf, -1, p.B, p.B);
            generate_exp_randomness<T, BW, smallBW, F, K_INT, 2>(p0_buf, p1_buf, -1, p.B, p.B);
            generate_reciprocal_non_extend_randomness<T, BW, smallBW, F, K_INT, 1>(p0_buf, p1_buf, -1, -1, 2 * p.B);
            FixTensor<T, smallBW, F, K_INT, 2> r_y_m(2 * p.B, p.B);
            T* val = rg.template randomGE<T>(2 * p.B, smallBW);
            for(int i = 0; i < 2 * p.B; i++){
                for(int j = 0; j < p.B; j++){
                    r_y_m(i, j) = Fix<T, smallBW, F, K_INT>(val[i]);
                }
            }
            delete[] val;
            generate_elementwise_mul_randomness<T, BW, smallBW, F, K_INT, 2, Eigen::RowMajor>(p0_buf, p1_buf, -1, 2 * p.B, p.B, nullptr, &r_y_m);
            generate_zero_extend_randomness<T, BW, smallBW, F, K_INT, 2>(p0_buf, p1_buf, -1, 2 * p.B, p.B);
            generate_zero_extend_randomness<T, BW, smallBW, F, K_INT, 2>(p0_buf, p1_buf, -1, p.B, p.B);

            generate_elementwise_mul_randomness<T, BW, smallBW, F, K_INT, 2, Eigen::RowMajor>(p0_buf, p1_buf, -1, p.B, p.B, &r_x_m_tensor, nullptr);
            generate_elementwise_mul_randomness<T, BW, smallBW, F, K_INT, 1, Eigen::RowMajor>(p0_buf, p1_buf, -1, 1, 1);
            generate_zero_extend_randomness<T, BW, smallBW, F, K_INT, 1>(p0_buf, p1_buf, -1, 1, 1);
            generate_zero_extend_randomness<T, BW, smallBW, F, K_INT, 1>(p0_buf, p1_buf, -1, 1, 1);
        }

        FixTensor<T, BW, F, K_INT, 2> backward(const FixTensor<T, smallBW, F, K_INT, 2>& input){
            // 1. tempearture = exp(lambda) shape: 1
            this->temperature = exp_scalar_without_extend<T, BW, smallBW, F, K_INT>(lambda, randomness.exp_scalar_randomness);
            
            // 2. inp_mul_temp = (input-ones) * temperature shape: B * B
            FixTensor<T, smallBW, F, K_INT, 2> ones(p.B, p.B);
            ones.setConstant(Fix<T, smallBW, F, K_INT>(1.0));
            FixTensor<T, smallBW, F, K_INT, 2> temperature_tensor(p.B, p.B);
            for(int i = 0; i < p.B; i++){
                for(int j = 0; j < p.B; j++){
                    temperature_tensor(i, j) = this->temperature;
                }
            }
            this->input_rec = input + randomness.elementwise_mul_randomness_temperature.r_x_m;
            this->input_rec = reconstruct_tensor(this->input_rec);
            auto input_minus_ones = this->input_rec - ones;
            FixTensor<T, BW, F, K_INT, 2> inp_mul_temp = elementwise_mul_opt<T, BW, smallBW, F, K_INT, 2, Eigen::RowMajor>(input_minus_ones, temperature_tensor, randomness.elementwise_mul_randomness_temperature, true);
            FixTensor<T, smallBW, F, K_INT, 2> inp_mul_temp_m = truncate_reduce_tensor(inp_mul_temp);
            FixTensor<T, BW, F, K_INT, 2> inp_mul_temp_extended = zero_extend_tensor<T, BW, smallBW, F, K_INT, 2>(inp_mul_temp_m, randomness.zero_extend_randomness_temperature);
            
            // 3. negative exp: shape: B * B
            // all exp of the input
            FixTensor<T, smallBW, F, K_INT, 2> exp_inp_mul_temp_m = exp_tensor_non_extend(inp_mul_temp_extended, randomness.exp_randomness);
            FixTensor<T, BW, F, K_INT, 2> exp_inp_mul_temp = zero_extend_tensor<T, BW, smallBW, F, K_INT, 2>(exp_inp_mul_temp_m, randomness.exp_randomness.zero_extend_randomness);

            // 4. reciprocal of exp sum row && col: shape: 2 * B
            FixTensor<T, BW, F, K_INT, 1> exp_sum = sum_reduce_tensor<1, T, BW, F, K_INT, Eigen::RowMajor>(exp_inp_mul_temp);
            FixTensor<T, BW, F, K_INT, 1> exp_sum_T = sum_reduce_tensor<0, T, BW, F, K_INT, Eigen::RowMajor>(exp_inp_mul_temp);
            // sum reduce result
            FixTensor<T, BW, F, K_INT, 1> exp_sum_concat(2 * p.B);
            for(int i = 0; i < p.B; i++){
                exp_sum_concat(i) = exp_sum(i) + (mpc_instance->party == 0 ? Fix<T, BW, F, K_INT>(0.1) : Fix<T, BW, F, K_INT>(0.0));
                exp_sum_concat(i + p.B) = exp_sum_T(i) + (mpc_instance->party == 0 ? Fix<T, BW, F, K_INT>(0.1) : Fix<T, BW, F, K_INT>(0.0));
            }
            FixTensor<T, smallBW, F, K_INT, 2> exp_inp_concat(2 * p.B, p.B);
            for(int i = 0; i < p.B; i++){
                for(int j = 0; j < p.B; j++){
                    exp_inp_concat(i, j) = exp_inp_mul_temp_m(i, j);
                    exp_inp_concat(i + p.B, j) = exp_inp_mul_temp_m(i, j);
                }
            }
            // 5. reciprocal of exp sum row && col
            // reciprocal_exp_inp_mul_temp shape: (2*B)
            FixTensor<T, smallBW, F, K_INT, 1> reciprocal_exp_inp_mul_temp = reciprocal_tensor_non_extend<T, BW, smallBW, F, K_INT, 1>(exp_sum_concat, randomness.reciprocal_randomness);
            // 6. reconstruct and then broadcast reciprocal to row and col
            for(int i = 0; i < 2 * p.B; i++){
                reciprocal_exp_inp_mul_temp(i) = reciprocal_exp_inp_mul_temp(i) + randomness.elementwise_mul_randomness.r_y_m(i, 0);
            }
            FixTensor<T, smallBW, F, K_INT, 1> reciprocal_sum_rec = reconstruct_tensor(reciprocal_exp_inp_mul_temp);
            FixTensor<T, smallBW, F, K_INT, 2> reciprocal_sum_broadcast_rec(2 * p.B, p.B);
            for(int i = 0; i < 2 * p.B; i++){
                for(int j = 0; j < p.B; j++){
                    reciprocal_sum_broadcast_rec(i, j) = reciprocal_sum_rec(i);
                }
            }
            
            // 5. softmax = exp_inp_mul_temp * reciprocal_sum_broadcast_rec shape: 2 * B * B
            FixTensor<T, BW, F, K_INT, 2> softmax = elementwise_mul_opt<T, BW, smallBW, F, K_INT, 2, Eigen::RowMajor>(exp_inp_concat, reciprocal_sum_broadcast_rec, randomness.elementwise_mul_randomness, false, true);
            FixTensor<T, smallBW, F, K_INT, 2> softmax_m = truncate_reduce_tensor(softmax);
            softmax = zero_extend_tensor<T, BW, smallBW, F, K_INT, 2>(softmax_m, randomness.zero_extend_randomness);
            FixTensor<T, BW, F, K_INT, 2> loss(p.B, p.B);
            for(int i = 0; i < p.B; i++){
                for(int j = 0; j < p.B; j++){
                    loss(i, j) = softmax(i, j) + softmax(i + p.B, j);
                    if(i == j){
                        if (mpc_instance->party == 0){
                            loss(i, j) = loss(i, j) - Fix<T, BW, F, K_INT>(2.0);
                        }
                    }
                }
            }
            FixTensor<T, BW, F, K_INT, 2> constant_n(p.B, p.B);
            constant_n.setConstant(Fix<T, BW, F, K_INT>(0.5));
            loss = loss * constant_n;
            FixTensor<T, smallBW, F, K_INT, 2> loss_m = truncate_reduce_tensor(loss);
            this->loss_m = loss_m;
            loss = zero_extend_tensor<T, BW, smallBW, F, K_INT, 2>(loss_m, randomness.zero_extend_randomness_loss, false);
            return loss;
        }

        void update(){
            auto loss_m_rec = reconstruct_tensor(this->loss_m);
            FixTensor<T, BW, F, K_INT, 2> dtao_tensor = elementwise_mul_opt<T, BW, smallBW, F, K_INT, 2, Eigen::RowMajor>(this->input_rec, this->loss_m, randomness.elementwise_mul_randomness_update, true, false);
            FixTensor<T, smallBW, F, K_INT, 2> dtao_m = truncate_reduce_tensor(dtao_tensor);
            auto dtao_m_rec = reconstruct_tensor(dtao_m);
            FixTensor<T, smallBW, F, K_INT, 1> dtao(1);
            FixTensor<T, smallBW, F, K_INT, 1> temperature_tensor(1);
            for(int i = 0; i < p.B; i++){
                for(int j = 0; j < p.B; j++){
                    dtao(0) = dtao(0) + dtao_m(i, j);
                }
            }
            temperature_tensor(0) = this->temperature;
        
            FixTensor<T, BW, F, K_INT, 1> dlambda = elementwise_mul_opt<T, BW, smallBW, F, K_INT, 1, Eigen::RowMajor>(dtao, temperature_tensor, randomness.elementwise_mul_randomness_update_lambda);
            FixTensor<T, smallBW, F, K_INT, 1> dlambda_m = truncate_reduce_tensor(dlambda);
            dlambda = zero_extend_tensor<T, BW, smallBW, F, K_INT, 1>(dlambda_m, randomness.zero_extend_randomness_update_dlambda[0]);
            dlambda = dlambda * Fix<T, BW, F, K_INT>(LR);
            dlambda_m = truncate_reduce_tensor(dlambda);
            dlambda = zero_extend_tensor<T, BW, smallBW, F, K_INT, 1>(dlambda_m, randomness.zero_extend_randomness_update_dlambda[1]);
            this->lambda = this->lambda - dlambda(0);
            auto lambda_rec = reconstruct(this->lambda);
            if(mpc_instance->party == 0){
                std::cout << "lambda_rec: " << lambda_rec << std::endl;
            }
            return;
        }
};