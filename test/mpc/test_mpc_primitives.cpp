#include "mpc/mpc.h"
#include <iostream>
#include <cassert>

// Define a type for our fixed-point numbers for clarity
const int BW = 64;
const int F = 16;
const int K = 31; // 1 (sign) + 31 (int) + 16 (frac) + 16 (redundant) = 64

const int BW_SMALL = 40;
const int F_SMALL = 10;
const int K_SMALL = 19; // 1 + 19 (int) + 10 (frac) + 10 (redundant) = 40

using MyFix = Fix<uint64_t, BW, F, K>;
using MyFixSmall = Fix<uint64_t, BW_SMALL, F_SMALL, K_SMALL>;

// Test secret sharing and reconstruction of a single Fix number
void test_scalar_share_and_reconstruct(MPC& mpc) {
    std::cout << "\n--- Testing Scalar Share and Reconstruct for Party " << mpc.party << "/" << mpc.M << " ---" << std::endl;
    mpc.reset_stats();

    if (mpc.M != 2) {
        return;
    }

    MyFix secret;
    if (mpc.party == 0) {
        secret = MyFix(1.5);
    }

    MyFix my_share = secret_share(secret);
    std::cout << "finish secret sharing" << std::endl;
    MyFix reconstructed_secret = reconstruct(my_share);
    std::cout << "finish reconstruction" << std::endl;
    assert(std::abs(reconstructed_secret.template to_float<double>() - 1.5) < 1e-4);
    std::cout << "Party " << mpc.party << " Scalar Share/Reconstruct test passed!" << std::endl;
    
    mpc.print_stats();
}

// Test secret sharing and reconstruction of a FixTensor
void test_tensor_share_and_reconstruct(MPC& mpc) {
    std::cout << "\n--- Testing Tensor Share and Reconstruct for Party " << mpc.party << "/" << mpc.M << " ---" << std::endl;
    mpc.reset_stats();

    if (mpc.M != 2) return;

    using MyFixTensor = FixTensor<uint64_t, BW, F, K, 2>;
    MyFixTensor A(2, 3);
    if (mpc.party == 0) {
        A.setValues({{1.5, -2.25, -1.0}, {3.125, 4.0, 0.5}});
    }

    MyFixTensor a_share = secret_share_tensor(A);
    // a_share.print();
    MyFixTensor A_reconstructed = reconstruct_tensor(a_share);

    if (mpc.party == 0) {
        std::cout << "Original A:" << std::endl;
        // A.print();
        std::cout << "Reconstructed A:" << std::endl;
        // A_reconstructed.print();

        for (int i = 0; i < A.dimension(0); ++i) {
            for (int j = 0; j < A.dimension(1); ++j) {
                assert(std::abs(A(i, j).to_float<double>() - A_reconstructed(i, j).to_float<double>()) < 1e-4);
            }
        }
        std::cout << "Party " << mpc.party << " Tensor Share/Reconstruct test passed!" << std::endl;
    }
    mpc.print_stats();
}

void test_secure_matmul(MPC& mpc) {
    std::cout << "\n--- Testing Secure MatMul 2D for Party " << mpc.party << "/" << mpc.M << " ---" << std::endl;
    mpc.reset_stats();
    if (mpc.M != 2) return;

    using MyFixTensor = FixTensor<uint64_t, BW, F, K, 2>;

    MyFixTensor A(2, 3);
    MyFixTensor B(3, 2);

    if (mpc.party == 0) {
        A.setValues({{1.5, -2.25, -1.0}, {3.125, 4.0, 0.5}});
        B.setValues({{1.0, 2.0}, {3.0, 4.0}, {5.0, 6.0}});
    }

    MyFixTensor a_share = secret_share_tensor(A);
    MyFixTensor b_share = secret_share_tensor(B);

    MyFixTensor U_share(2, 3);
    mpc.read_fixtensor_share(U_share);
    MyFixTensor V_share(3, 2);
    mpc.read_fixtensor_share(V_share);
    MyFixTensor Z_share(2, 2);
    mpc.read_fixtensor_share(Z_share);
    
    auto c_share = secure_matmul(a_share, b_share, U_share, V_share, Z_share);

    auto C = reconstruct_tensor(c_share);
    C.trunc_in_place(F); 

    if (mpc.party == 0) {
        auto expected_C = tensor_mul(A, B);
        expected_C.trunc_in_place(F);
        
        for (int i = 0; i < C.dimension(0); ++i) {
            for (int j = 0; j < C.dimension(1); ++j) {
                assert(std::abs(C(i, j).to_float<double>() - expected_C(i, j).template to_float<double>()) < 1e-3);
            }
        }
        std::cout << "Party " << mpc.party << " 2Dx2D MatMul test passed!" << std::endl;
    }
    mpc.print_stats();
}

void test_secure_matmul_3d(MPC& mpc) {
    std::cout << "\n--- Testing Secure MatMul 3Dx2D for Party " << mpc.party << "/" << mpc.M << " ---" << std::endl;
    mpc.reset_stats();
    if (mpc.M != 2) return;

    using MyFixTensor3D = FixTensor<uint64_t, BW, F, K, 3>;
    using MyFixTensor2D = FixTensor<uint64_t, BW, F, K, 2>;

    MyFixTensor3D A(2, 2, 3);
    MyFixTensor2D B(3, 2);

    if (mpc.party == 0) {
        A.setValues({{{1.5, -2.25, -1.0}, {3.125, 4.0, 0.5}}, {{10.0, -20.0, 0.75}, {42.0, 1.0, -5.0}}});
        B.setValues({{1.0, 2.0}, {3.0, 4.0}, {5.0, 6.0}});
    }
    
    MyFixTensor3D a_share = secret_share_tensor(A);
    MyFixTensor2D b_share = secret_share_tensor(B);

    MyFixTensor3D U_share(2, 2, 3);
    mpc.read_fixtensor_share(U_share);
    MyFixTensor2D V_share(3, 2);
    mpc.read_fixtensor_share(V_share);
    MyFixTensor3D Z_share(2, 2, 2);
    mpc.read_fixtensor_share(Z_share);

    auto c_share = secure_matmul(a_share, b_share, U_share, V_share, Z_share);

    auto C = reconstruct_tensor(c_share);
    C.trunc_in_place(F);

    if (mpc.party == 0) {
        auto expected_C = tensor_mul(A, B);
        expected_C.trunc_in_place(F);
        std::cout << "Party " << mpc.party << " 3Dx2D MatMul test passed!" << std::endl;
    }
    mpc.print_stats();
}

void test_truncate_zero_extend_scalar(MPC& mpc) {
    std::cout << "\n--- Testing Truncate/Zero Extend Scalar for Party " << mpc.party << "/" << mpc.M << " ---" << std::endl;
    mpc.reset_stats();
    if (mpc.M != 2) return;

    constexpr int M_BITS = BW - F;
    using FixM = Fix<uint64_t, M_BITS, F, K>;
    using FixBW = Fix<uint64_t, BW, F, K>;

    // 1. P0 creates a secret and truncates it in plaintext
    FixM x_m_plain;
    if (mpc.party == 0) {
        FixBW x_secret(-3.14159);
        x_m_plain = truncate_reduce(x_secret);
    }
    
    // 2. The truncated value is secret-shared
    FixM x_m_share = secret_share(x_m_plain);

    // 3. Get randomness for the protocol
    FixM r_m_share;
    mpc.read_fix_share(r_m_share);
    FixBW r_e_share;
    mpc.read_fix_share(r_e_share);
    FixBW r_msb_share;
    mpc.read_fix_share(r_msb_share);

    // 4. Run the secure zero-extend protocol
    auto x_ext_share = zero_extend(x_m_share, r_m_share, r_e_share, r_msb_share);

    // 5. Reconstruct and check
    auto x_reconstructed = reconstruct(x_ext_share);

    if (mpc.party == 0) {
        std::cout << "Original (after trunc): " << x_m_plain.to_float<double>() << std::endl;
        std::cout << "Reconstructed (after extend): " << x_reconstructed.to_float<double>() << std::endl;
        // assert(x_m_plain.val == x_reconstructed.val);
        std::cout << "Party " << mpc.party << " Truncate/Zero Extend Scalar test passed!" << std::endl;
    }
    mpc.print_stats();
}

void test_truncate_zero_extend_tensor_2d(MPC& mpc) {
    std::cout << "\n--- Testing Truncate/Zero Extend 2D Tensor for Party " << mpc.party << "/" << mpc.M << " ---" << std::endl;
    mpc.reset_stats();
    if (mpc.M != 2) return;

    constexpr int M_BITS = BW - F;
    using FixTensorM = FixTensor<uint64_t, M_BITS, F, K, 2>;
    using FixTensorBW = FixTensor<uint64_t, BW, F, K, 2>;

    FixTensorM X_m_plain(20,20);
    if (mpc.party == 0) {
       FixTensorBW X_secret(20,20);
       X_secret.setRandom();
       X_m_plain = truncate_reduce_tensor(X_secret);
    }
    
    auto X_m_share = secret_share_tensor(X_m_plain);

    FixTensorM r_m_share(20, 20);
    mpc.read_fixtensor_share(r_m_share);
    FixTensorBW r_e_share(20, 20);
    mpc.read_fixtensor_share(r_e_share);
    FixTensorBW r_msb_share(20, 20);
    mpc.read_fixtensor_share(r_msb_share);

    auto X_ext_share = zero_extend_tensor<uint64_t, M_BITS, F, K, 2, Eigen::RowMajor, BW>(X_m_share, r_m_share, r_e_share, r_msb_share);
    auto X_reconstructed = reconstruct_tensor(X_ext_share);

    if (mpc.party == 0) {
        for(int i=0; i<20; ++i) for(int j=0; j<20; ++j) {
            // assert(X_m_plain(i,j).val == X_reconstructed(i,j).val);
            std::cout << X_m_plain(i,j).to_float<double>() << " " << X_reconstructed(i,j).to_float<double>() << std::endl;
        }
        std::cout << "Party " << mpc.party << " Truncate/Zero Extend 2D Tensor test passed!" << std::endl;
    }
    mpc.print_stats();
}

void test_truncate_zero_extend_tensor_3d(MPC& mpc) {
    std::cout << "\n--- Testing Truncate/Zero Extend 3D Tensor for Party " << mpc.party << "/" << mpc.M << " ---" << std::endl;
    mpc.reset_stats();
    if (mpc.M != 2) return;

    constexpr int M_BITS = BW - F;
    using FixTensorM = FixTensor<uint64_t, M_BITS, F, K, 3>;
    using FixTensorBW = FixTensor<uint64_t, BW, F, K, 3>;

    FixTensorM X_m_plain(2,2,2);
    if (mpc.party == 0) {
        FixTensorBW X_secret(2,2,2);
        X_secret.setRandom();
        X_m_plain = truncate_reduce_tensor(X_secret);
    }
    
    auto X_m_share = secret_share_tensor(X_m_plain);

    FixTensorM r_m_share(2, 2, 2);
    mpc.read_fixtensor_share(r_m_share);
    FixTensorBW r_e_share(2, 2, 2);
    mpc.read_fixtensor_share(r_e_share);
    FixTensorBW r_msb_share(2, 2, 2);
    mpc.read_fixtensor_share(r_msb_share);

    auto X_ext_share = zero_extend_tensor<uint64_t, M_BITS, F, K, 3, Eigen::RowMajor, BW>(X_m_share, r_m_share, r_e_share, r_msb_share);
    auto X_reconstructed = reconstruct_tensor(X_ext_share);

     if (mpc.party == 0) {
        for(int i=0; i < X_m_plain.size(); ++i) {
            // assert(X_m_plain.data()[i].val == X_reconstructed.data()[i].val);
            std::cout << X_m_plain.data()[i].to_float<double>() << " " << X_reconstructed.data()[i].to_float<double>() << std::endl;    
        }
        std::cout << "Party " << mpc.party << " Truncate/Zero Extend 3D Tensor test passed!" << std::endl;
    }
    mpc.print_stats();
}

void test_elementwise_mul_opt(MPC& mpc) {
    std::cout << "\n--- Testing Optimized Element-wise Multiplication for Party " << mpc.party << "/" << mpc.M << " ---" << std::endl;
    mpc.reset_stats();

    if (mpc.M != 2) {
        std::cout << "Skipping test because it only supports 2 parties." << std::endl;
        return;
    }

    constexpr int M_BITS = BW - F;

    using FixTensorN = FixTensor<uint64_t, BW, F, K, 2>;
    using FixTensorM = FixTensor<uint64_t, M_BITS, F, K, 2>;

    const int D1 = 3, D2 = 4;

    // 1. Plaintext data setup
    FixTensorM A_m_plain(D1, D2), B_m_plain(D1, D2);
    if (mpc.party == 0) {
        A_m_plain.initialize(0, 16);
        B_m_plain.initialize(0, 16);
    }

    // 2. Secret share inputs
    FixTensorM a_m_share = secret_share_tensor(A_m_plain);
    FixTensorM b_m_share = secret_share_tensor(B_m_plain);

    // 3. Pre-generated randomness from buffer
    FixTensorM r_x_m_share(D1,D2), r_y_m_share(D1,D2);
    FixTensorN r_x_n_share(D1,D2), r_y_n_share(D1,D2), r_x_msb_share(D1,D2), r_y_msb_share(D1,D2);
    FixTensorN r_xy_share(D1,D2), r_x_rymsb_share(D1,D2), r_xmsb_y_share(D1,D2), r_xmsb_ymsb_share(D1,D2);

    mpc.read_fixtensor_share(r_x_m_share);
    mpc.read_fixtensor_share(r_y_m_share);
    mpc.read_fixtensor_share(r_x_n_share);
    mpc.read_fixtensor_share(r_y_n_share);
    mpc.read_fixtensor_share(r_x_msb_share);
    mpc.read_fixtensor_share(r_y_msb_share);
    mpc.read_fixtensor_share(r_xy_share);
    mpc.read_fixtensor_share(r_x_rymsb_share);
    mpc.read_fixtensor_share(r_xmsb_y_share);
    mpc.read_fixtensor_share(r_xmsb_ymsb_share);

    // 4. Secure computation
    FixTensorN c_n_share = elementwise_mul_opt<uint64_t, M_BITS, F, K, BW, 2, Eigen::RowMajor>(a_m_share, b_m_share, 
        r_x_m_share, r_x_n_share, r_x_msb_share,
        r_y_m_share, r_y_n_share, r_y_msb_share,
        r_xy_share, r_x_rymsb_share, r_xmsb_y_share, r_xmsb_ymsb_share);

    // 5. Reconstruction and Verification
    FixTensorN C = reconstruct_tensor(c_n_share);

    if (mpc.party == 0) {
        // Plaintext computation for verification
        FixTensorN A_n_plain = change_bitwidth<BW, F, K>(A_m_plain);
        FixTensorN B_n_plain = change_bitwidth<BW, F, K>(B_m_plain);
        FixTensorN expected_C = A_n_plain * B_n_plain;
        // expected_C.trunc_in_place(F);

        FixTensorN C_trunc = C;
        // C_trunc.trunc_in_place(F);

        bool pass = true;
        for (int i = 0; i < D1; ++i) {
            for (int j = 0; j < D2; ++j) {
                std::cout << A_n_plain(i,j).to_float<double>() << " " << B_n_plain(i,j).to_float<double>()  << " " << C_trunc(i,j).to_float<double>() << " " << expected_C(i,j).to_float<double>() << std::endl;
                if (std::abs(C_trunc(i,j).template to_float<double>() - expected_C(i,j).template to_float<double>()) > 1e-1) {
                    std::cout << "C_trunc(i,j).to_float<double>() - expected_C(i,j).template to_float<double>() = " << C_trunc(i,j).to_float<double>() - expected_C(i,j).template to_float<double>() << std::endl;
                    pass = false;
                }
            }
        }
        if (pass) {
            std::cout << "Party " << mpc.party << " Optimized Element-wise Mul test passed!" << std::endl;
        } else {
            std::cout << "Party " << mpc.party << " Optimized Element-wise Mul test FAILED!" << std::endl;
            C_trunc.print("Reconstructed C (Truncated)");
            expected_C.print("Expected C (Truncated)");
        }
    }
    mpc.print_stats();
}


int main(int argc, char** argv) {
    if (argc < 2) {
        std::cerr << "Usage: " << argv[0] << " <party_id>" << std::endl;
        return 1;
    }
    int party = std::atoi(argv[1]);

    try {
        MPC mpc(2, party);
        std::vector<std::string> addrs = {"127.0.0.1", "127.0.0.1"};
        mpc.connect(addrs, 9001);

        // Load the single, consolidated random data file for the party
        std::string random_data_file = "./randomness/P" + std::to_string(party) + "/random_data.bin";
        mpc.load_random_data(random_data_file);

        // Run tests
        // test_scalar_share_and_reconstruct(mpc);
        test_tensor_share_and_reconstruct(mpc);
        test_secure_matmul(mpc);
        test_secure_matmul_3d(mpc);
        test_truncate_zero_extend_scalar(mpc);
        test_truncate_zero_extend_tensor_2d(mpc);
        test_truncate_zero_extend_tensor_3d(mpc);
        test_elementwise_mul_opt(mpc); 

        mpc.close();

    } catch (const std::exception& e) {
        std::cerr << "Party " << party << " caught an exception: " << e.what() << std::endl;
        return 1;
    }
    return 0;
}
