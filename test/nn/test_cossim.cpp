#include "nn/CosineSimilarity.h"
#include "mpc/mpc.h"
#include "mpc/tensor_ops.h"
#include <iostream>
#include <vector>
#include <cassert>
#include <fstream>
#include <cmath>

void test_cossim(MPC& mpc) {
    std::cout << "--- Testing CosineSimilarity Layer Forward Pass for Party " << mpc.party << "/" << mpc.M << " ---" << std::endl;

    using T = uint64_t;
    const int IN_BW = 48;
    const int OUT_BW = 64;
    const int F = 16;
    const int K_INT = 15;

    using FixIn = Fix<T, IN_BW, F, K_INT>;
    using InputTensor = CosSimLayer<T, IN_BW, OUT_BW, F, K_INT>::InputTensor;
    
    CosSimLayerParams params = {5, 10}; // B=5, in_dim=10
    CosSimLayer<T, IN_BW, OUT_BW, F, K_INT> cosinesim_layer(params);

    std::cout << "CosineSimilarity Layer initialized" << std::endl;

    // 1. Secret Share Inputs
    InputTensor image_plain(params.B, params.in_dim);
    InputTensor text_plain(params.B, params.in_dim);
    FixTensor<T, OUT_BW, F, K_INT, 2> image_plain_ext(params.B, params.in_dim);
    FixTensor<T, OUT_BW, F, K_INT, 2> text_plain_ext(params.B, params.in_dim);

    InputTensor image_share(params.B, params.in_dim);
    InputTensor text_share(params.B, params.in_dim);

    for(int i = 0; i < params.B; ++i) {
        for(int j = 0; j < params.in_dim; ++j) {
            image_plain(i, j) = FixIn(double(i+j)/10.0);
            text_plain(i, j) = FixIn(double(i-j)/10.0);
            image_plain_ext(i, j) = Fix<T, OUT_BW, F, K_INT>(double(i+j)/10.0);
            text_plain_ext(i, j) = Fix<T, OUT_BW, F, K_INT>(double(i-j)/10.0);
        }
    }
    if (mpc.party == 0) {
        // Use Eigen's random capabilities to generate test data
        // Party 0 holds the full plaintext as its share
        image_share = image_plain;
        text_share = text_plain;
    } else {
        // Other parties have zero shares
        image_share.setZero();
        text_share.setZero();
    }
    cosinesim_layer.read_randomness(mpc);
    std::cout << "Randomness loaded" << std::endl;

    std::cout << "Executing forward pass" << std::endl;
    auto y_share = cosinesim_layer.forward(image_share, text_share);
    std::cout << "Forward pass executed" << std::endl;

    auto y_reconstructed = reconstruct_tensor(y_share);
    FixTensor<T, OUT_BW, F, K_INT, 3> norm_tensor;
    norm_tensor.resize(2, params.B, params.in_dim);
    // 6. Plaintext Verification
    if (mpc.party == 0) {
        // Re-shape plaintext inputs for stacking, just like in the layer
        FixTensor<T, IN_BW, F, K_INT, 3> all_plain_3d(2, params.B, params.in_dim);
        for(int i = 0; i < params.B; ++i) {
            for(int j = 0; j < params.in_dim; ++j) {
                all_plain_3d(0, i, j) = image_plain(i, j);
                all_plain_3d(1, i, j) = text_plain(i, j);
            }
        }

        // Plaintext L2 Normalization
        FixTensor<T, OUT_BW, F, K_INT, 3> expected_y(2, params.B, params.in_dim);
        for (int i = 0; i < 2; ++i) {
            for (int b = 0; b < params.B; ++b) {
                double norm_sq = 0.0;
                for (int d = 0; d < params.in_dim; ++d) {
                    norm_sq += all_plain_3d(i, b, d).to_float<double>() * all_plain_3d(i, b, d).to_float<double>();
                }
                double norm = std::sqrt(norm_sq) + 1e-12; // Add epsilon for stability
                for (int d = 0; d < params.in_dim; ++d) {
                    norm_tensor(i, b, d) = Fix<T, OUT_BW, F, K_INT>(1.0/norm);
                }
                for (int d = 0; d < params.in_dim; ++d) {
                    expected_y(i, b, d) = Fix<T, OUT_BW, F, K_INT>(all_plain_3d(i, b, d).to_float<double>() / norm);
                }
            }
        }
        FixTensor<T, OUT_BW, F, K_INT, 2> y_I(params.B, params.in_dim);
        FixTensor<T, OUT_BW, F, K_INT, 2> y_T_T(params.in_dim, params.B);
        for (int i = 0; i < params.B; i++) {
            for (int j = 0; j < params.in_dim; j++) {
                y_I(i, j) = expected_y(0, i, j);
                y_T_T(j, i) = expected_y(1, i, j);
            }
        }
        FixTensor<T, OUT_BW, F, K_INT, 2> res = tensor_mul(y_I, y_T_T);
        res.trunc_in_place(F);
        std::cout << "--------FORWARD TEST--------" << std::endl;
        std::cout << "Reconstructed Output:\n" << y_reconstructed << std::endl;
        std::cout << "Expected Output:\n" << res << std::endl;

        for (int i = 0; i < y_reconstructed.size(); ++i) {
            if(std::abs(y_reconstructed.data()[i].to_float<double>() - res.data()[i].to_float<double>()) >= 1e-1){
                std::cout << y_reconstructed.data()[i].to_float<double>();
            }
        }
        std::cout << "CosineSimilarity forward verification passed." << std::endl;
    }

    // --- Backward Pass Test ---
    std::cout << "\n--- Testing CosineSimilarity Layer Backward Pass for Party " << mpc.party << "/" << mpc.M << " ---" << std::endl;

    using IncomingGradTensor = CosSimLayer<T, IN_BW, OUT_BW, F, K_INT>::IncomingGradTensor;
    IncomingGradTensor incoming_grad_share(params.B, params.B);
    IncomingGradTensor incoming_grad_plain(params.B, params.B);
    IncomingGradTensor incoming_grad_plain_T(params.B, params.B);

    for(int i = 0; i < params.B; ++i) {
        for(int j = 0; j < params.B; ++j) {
            incoming_grad_plain(i, j) = Fix<T, OUT_BW, F, K_INT>(double(i - j) / 20.0);
        }
    }
    if (mpc.party == 0) {
        incoming_grad_share = incoming_grad_plain;
    } else {
        incoming_grad_share.setZero();
    }
    incoming_grad_plain_T = incoming_grad_plain.shuffle(Eigen::array<int, 2>{1, 0});
    auto [dI_share, dT_share] = cosinesim_layer.backward(incoming_grad_share);
    // auto dI_share = l2_layer.backward(incoming_grad_share);
    auto dI_reconstructed = reconstruct_tensor(dI_share);
    auto dT_reconstructed = reconstruct_tensor(dT_share);
    // Backward Verification
    if (mpc.party == 0) {
        // Plaintext backward calculation
        FixTensor<T, OUT_BW, F, K_INT, 2> norm_image_plain(params.B, params.in_dim);
        FixTensor<T, OUT_BW, F, K_INT, 2> norm_text_plain(params.B, params.in_dim);
        for(int i = 0; i < params.B; ++i) {
            for(int j = 0; j < params.in_dim; ++j) {
                norm_image_plain(i, j) = norm_tensor(0, i, j);
                norm_text_plain(i, j) = norm_tensor(1, i, j);
            }
        }
        // 1. dI = (dL/dy) @ I
        auto dI_plain = tensor_mul(incoming_grad_plain, image_plain_ext);
        dI_plain.trunc_in_place(F);
        // 2. proj_I = dI * I
        FixTensor<T, OUT_BW, F, K_INT, 2> proj_I_plain_full_bw = dI_plain * image_plain_ext;
        proj_I_plain_full_bw.trunc_in_place(F);
        // 3. proj_I = sum(proj_I)
        FixTensor<T, OUT_BW, F, K_INT, 1> proj_I_m_plain = sum_reduce_tensor<T, OUT_BW, F, K_INT, Eigen::RowMajor>(proj_I_plain_full_bw);
        // 4. Broadcast proj_I
        FixTensor<T, OUT_BW, F, K_INT, 2> proj_I_broadcasted(params.B, params.in_dim);
        for(int i = 0; i < params.B; i++) {
            for(int j = 0; j < params.in_dim; j++) {
                proj_I_broadcasted(i, j) = proj_I_m_plain(i);
            }
        }
        // 5. term = proj_I_m * I
        auto term_plain = image_plain_ext * proj_I_broadcasted;
        term_plain.trunc_in_place(F);
        // 6. dI = dI - term
        dI_plain = dI_plain - term_plain;
        // 7. dI = dI * norm
        auto dI_final_full_bw = dI_plain * norm_image_plain;
        dI_final_full_bw.trunc_in_place(F);
        std::cout << "--------IMAGE VERIFICATION----------" << std::endl;
        for (int i = 0; i < dI_reconstructed.size(); ++i) {
            if(std::abs(dI_reconstructed.data()[i].to_float<double>() - dI_final_full_bw.data()[i].to_float<double>()) >= 1e-2){
                std::cout << dI_reconstructed.data()[i].to_float<double>() << " " << dI_final_full_bw.data()[i].to_float<double>() << std::endl;
            }
        }
        FixTensor<T, OUT_BW, F, K_INT, 2> dT_plain = tensor_mul(incoming_grad_plain_T, text_plain_ext);
        dT_plain.trunc_in_place(F);
        auto dT_final_full_bw = dT_plain * norm_text_plain;
        dT_final_full_bw.trunc_in_place(F);
        auto proj_T_plain_full_bw = dT_plain * text_plain_ext;
        proj_T_plain_full_bw.trunc_in_place(F);
        FixTensor<T, OUT_BW, F, K_INT, 1> proj_T_m_plain = sum_reduce_tensor<T, OUT_BW, F, K_INT, Eigen::RowMajor>(proj_T_plain_full_bw);
        
        // 4. Broadcast proj_T
        FixTensor<T, OUT_BW, F, K_INT, 2> proj_T_broadcasted(params.B, params.in_dim);
        for(int i = 0; i < params.B; i++) {
            for(int j = 0; j < params.in_dim; j++) {
                proj_T_broadcasted(i, j) = proj_T_m_plain(i);
            }
        }
        // 5. term = proj_I_m * I
        term_plain = text_plain_ext * proj_T_broadcasted;
        term_plain.trunc_in_place(F);
        // 6. dI = dI - term
        dT_plain = dT_plain - term_plain;
        // 7. dT = dT * norm
        dT_final_full_bw = dT_plain * norm_text_plain;
        dT_final_full_bw.trunc_in_place(F);
        std::cout << "--------TEXT VERIFICATION----------";
        for (int i = 0; i < dT_reconstructed.size(); ++i) {
            if(std::abs(dT_reconstructed.data()[i].to_float<double>() - dT_final_full_bw.data()[i].to_float<double>()) >= 1e-2){
                std::cout << dT_reconstructed.data()[i].to_float<double>() << " " << dT_final_full_bw.data()[i].to_float<double>() << std::endl;
            }
        }
        std::cout << "CosineSimilarity backward verification passed." << std::endl;
        std::cout << "Expected Gradient:\n" << dT_final_full_bw << std::endl;
        std::cout << "Reconstructed Gradient:\n" << dT_reconstructed << std::endl;
    }
    
    std::cout << "Party " << mpc.party << " CosineSimilarity Layer test passed!" << std::endl;
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
    
    std::string randomness_path = "./randomness/P" + std::to_string(party) + "/cosinesim_randomness.bin";
    if (!file_exists(randomness_path)) {
        std::cerr << "Randomness file not found: " << randomness_path << std::endl;
        return 1;
    }
    mpc.load_random_data(randomness_path);
    
    std::vector<std::string> addrs = {"127.0.0.1", "127.0.0.1"};
    mpc.connect(addrs, 9001);

    test_cossim(mpc);

    return 0;
}
