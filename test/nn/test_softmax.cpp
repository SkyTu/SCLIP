#include "nn/Softmax.h"
#include "mpc/mpc.h"
#include "mpc/tensor_ops.h"
#include <iostream>
#include <vector>
#include <cassert>
#include <fstream>
#include <cmath>

void test_softmax(MPC& mpc) {
    std::cout << "--- Testing Softmax Layer for Party " << mpc.party << "/" << mpc.M << " ---" << std::endl;

    using T = uint64_t;
    const int BW = 64;
    const int smallBW = 44;
    const int F = 20;
    const int K_INT = 15;

    using FixIn = Fix<T, smallBW, F, K_INT>;
    using InputTensor = FixTensor<T, smallBW, F, K_INT, 2>;
    
    SoftmaxLayerParams params = {5}; // B=5
    SoftmaxLayer<T, BW, smallBW, F, K_INT> softmax_layer(params);

    std::cout << "Softmax Layer initialized" << std::endl;

    // 1. Secret Share Inputs
    InputTensor input_plain(params.B, params.B);
    InputTensor input_share(params.B, params.B);

    for(int i = 0; i < params.B; ++i) {
        for(int j = 0; j < params.B; ++j) {
            input_plain(i, j) = FixIn(double(i - j) / 5);
        }
    }

    if (mpc.party == 0) {
        input_share = input_plain;
    } else {
        input_share.setZero();
    }
    
    softmax_layer.read_randomness(mpc);
    std::cout << "Randomness loaded" << std::endl;

    std::cout << "Executing backward pass" << std::endl;
    FixTensor<T, BW, F, K_INT, 2> loss_share = softmax_layer.backward(input_share);
    std::cout << "Backward pass executed" << std::endl;

    auto loss_reconstructed = reconstruct_tensor(loss_share);

    // Plaintext Verification
    if (mpc.party == 0) {
        // 1. temperature = exp(lambda)
        double temperature = exp(2.6);
        using PlainTensorN = FixTensor<T, BW, F, K_INT, 2>;
        using FixN = Fix<T, BW, F, K_INT>;
        PlainTensorN input_plain_n(params.B, params.B);
        input_plain_n = change_bitwidth<BW, F, K_INT, T, smallBW, F, K_INT, 2, Eigen::RowMajor>(input_plain);
        std::cout << "input_plain_n: " << input_plain_n << std::endl;
        // 2. inp_mul_temp = (input-ones) * temperature
        PlainTensorN ones(params.B, params.B);
        ones.setConstant(FixN(1.0));
        PlainTensorN inp_mul_temp = (input_plain_n - ones) * FixN(temperature);
        inp_mul_temp.trunc_in_place(F);
        std::cout << "inp_mul_temp: " << inp_mul_temp << std::endl;
        // 3. negative exp
        PlainTensorN exp_inp_mul_temp(params.B, params.B);
        for(int i = 0; i < params.B; ++i) {
            for(int j = 0; j < params.B; ++j) {
                exp_inp_mul_temp(i, j) = FixN(std::exp(inp_mul_temp(i, j).to_float<double>()));
            }
        }
        std::cout << "exp_inp_mul_temp: " << exp_inp_mul_temp << std::endl;
        
        // 4. reciprocal of exp sum row && col
        FixTensor<T, BW, F, K_INT, 1> exp_sum_row = sum_reduce_tensor<1, T, BW, F, K_INT, Eigen::RowMajor>(exp_inp_mul_temp);
        FixTensor<T, BW, F, K_INT, 1> exp_sum_col = sum_reduce_tensor<0, T, BW, F, K_INT, Eigen::RowMajor>(exp_inp_mul_temp);

        for(int i = 0; i < params.B; i++){
            exp_sum_row(i) = exp_sum_row(i) + FixN(0.1);
            exp_sum_col(i) = exp_sum_col(i) + FixN(0.1);
        }

        // 5. softmax calculation
        PlainTensorN softmax_row(params.B, params.B);
        PlainTensorN softmax_col(params.B, params.B);
        FixTensor<T, BW, F, K_INT, 1> inverse_concat(2 * params.B);

        for(int i = 0; i < params.B; ++i) {
            FixN inv_row_sum = FixN(1.0 / (exp_sum_row(i).to_float<double>()));
            inverse_concat(i) = inv_row_sum;
            FixN inv_col_sum = FixN(1.0 / (exp_sum_col(i).to_float<double>()));
            inverse_concat(i + params.B) = inv_col_sum;
            for(int j = 0; j < params.B; ++j) {
                softmax_row(i, j) = exp_inp_mul_temp(i, j) * inv_row_sum;
                softmax_col(j, i) = exp_inp_mul_temp(j, i) * inv_col_sum;
            }
        }
        std::cout << "inverse_concat: " << inverse_concat << std::endl;
        
        softmax_row.trunc_in_place(F);
        softmax_col.trunc_in_place(F);

        std::cout << "softmax_row: " << softmax_row << std::endl;
        std::cout << "softmax_col: " << softmax_col << std::endl;
        // 6. loss calculation
        PlainTensorN loss_plain(params.B, params.B);
        for(int i = 0; i < params.B; i++){
            for(int j = 0; j < params.B; j++){
                loss_plain(i, j) = softmax_row(i, j) + softmax_col(i, j);
                if(i == j){
                    loss_plain(i, j) = loss_plain(i, j) - FixN(2.0);
                }
            }
        }
        ones.setConstant(FixIn(0.5));
        loss_plain = loss_plain * ones;
        loss_plain.trunc_in_place(F);
        // std::cout << "loss_plain: " << loss_plain << std::endl;
        std::cout << "--------BACKWARD TEST--------" << std::endl;
        std::cout << "Reconstructed Output:\n" << loss_reconstructed << std::endl;
        std::cout << "Expected Output:\n" << loss_plain << std::endl;

        for (int i = 0; i < loss_reconstructed.size(); ++i) {
            assert(std::abs(loss_reconstructed.data()[i].to_float<double>() - loss_plain.data()[i].to_float<double>()) < 3e-1);
        }
        std::cout << "Softmax backward verification passed." << std::endl;
    }
    
    std::cout << "Party " << mpc.party << " Softmax Layer test passed!" << std::endl;
    softmax_layer.update();
    std::cout << "Softmax Layer updated" << std::endl;
}

bool file_exists(const std::string& path) {
    std::ifstream file(path);
    return file.good();
}

int main(int argc, char** argv) {
    if (argc < 2) {
        std::cout << "Usage: " << argv[0] << " <party>" << std::endl;
        return 1;
    }
    int party = std::atoi(argv[1]);
    MPC mpc(2, party);
    
    std::string randomness_path = "./randomness/P" + std::to_string(party) + "/softmax_randomness.bin";
    if (!file_exists(randomness_path)) {
        std::cerr << "Randomness file not found: " << randomness_path << std::endl;
        return 1;
    }
    mpc.load_random_data(randomness_path);
    
    std::vector<std::string> addrs = {"127.0.0.1", "127.0.0.1"};
    mpc.connect(addrs, 9001);

    test_softmax(mpc);

    return 0;
}
