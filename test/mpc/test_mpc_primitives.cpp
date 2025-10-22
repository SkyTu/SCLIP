#include "mpc/mpc.h"
#include "mpc/truncate.h"
#include "mpc/elementwise_mul.h"
#include "mpc/square.h"
#include "mpc/matmul.h"
#include "mpc/secure_tensor_ops.h"
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

void test_reconstruct_parallel(MPC& mpc) {
    std::cout << "\n--- Testing Parallel Tensor Reconstruct for Party " << mpc.party << "/" << mpc.M << " ---" << std::endl;
    mpc.reset_stats();

    if (mpc.M != 2) return;

    using FixTensor2D = FixTensor<uint64_t, BW, F, K, 2>;
    using FixTensor3D = FixTensor<uint64_t, BW, F, K, 3>;

    FixTensor2D A_plain(2, 3);
    FixTensor3D B_plain(2, 2, 2);
    FixTensor2D C_plain(2, 2);

    if (mpc.party == 0) {
        A_plain.setValues({{1.1, 2.2, 3.3}, {4.4, 5.5, 6.6}});
        B_plain.setValues({{{-1.0, -2.0}, {-3.0, -4.0}}, {{-5.0, -6.0}, {-7.0, -8.0}}});
        C_plain.setValues({{1.0, 2.0}, {3.0, 4.0}});
    }

    // Create shares
    FixTensor2D a_share = secret_share_tensor(A_plain);
    FixTensor3D b_share = secret_share_tensor(B_plain);
    FixTensor2D c_share = secret_share_tensor(C_plain);

    // Reconstruct both tensors in parallel
    // After this call, a_share and b_share will hold the reconstructed values
    reconstruct_tensor_parallel(a_share, b_share, c_share);

    if (mpc.party == 0) {
        bool pass = true;
        for (int i = 0; i < A_plain.dimension(0); ++i) {
            for (int j = 0; j < A_plain.dimension(1); ++j) {
                if (std::abs(A_plain(i, j).to_float<double>() - a_share(i, j).to_float<double>()) > 1e-4) {
                    pass = false;
                }
            }
        }

        for (int i = 0; i < B_plain.dimension(0); ++i) {
            for (int j = 0; j < B_plain.dimension(1); ++j) {
                for (int k = 0; k < B_plain.dimension(2); ++k) {
                    if (std::abs(B_plain(i,j,k).to_float<double>() - b_share(i,j,k).to_float<double>()) > 1e-4) {
                        pass = false;
                    }
                }
            }
        }

        for (int i = 0; i < C_plain.dimension(0); ++i) {
            for (int j = 0; j < C_plain.dimension(1); ++j) {
                if (std::abs(C_plain(i, j).to_float<double>() - c_share(i, j).to_float<double>()) > 1e-4) {
                    pass = false;
                }
            }
        }
        
        if (pass) {
            std::cout << "Party " << mpc.party << " Parallel Tensor Reconstruct test passed!" << std::endl;
        } else {
            std::cout << "Party " << mpc.party << " Parallel Tensor Reconstruct test FAILED!" << std::endl;
        }
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

    MatmulRandomness<uint64_t, BW, F, K, 2, 2, 2> randomness = read_matmul_randomness<uint64_t, BW, F, K, 2, 2, 2>(mpc, 0, 2, 3, 2);
    
    auto c_share = secure_matmul(a_share, b_share, randomness);

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

    MatmulRandomness<uint64_t, BW, F, K, 3, 2, 3> randomness = read_matmul_randomness<uint64_t, BW, F, K, 3, 2, 3>(mpc, 2, 2, 3, 2);

    auto c_share = secure_matmul(a_share, b_share, randomness);

    auto C = reconstruct_tensor(c_share);
    C.trunc_in_place(F);

    bool pass = true;
    if (mpc.party == 0) {
        auto expected_C = tensor_mul(A, B);
        expected_C.trunc_in_place(F);

        for(int i=0; i<C.dimension(0); ++i) for(int j=0; j<C.dimension(1); ++j) for(int k=0; k<C.dimension(2); ++k) {
            if(std::abs(C(i,j,k).to_float<double>() - expected_C(i,j,k).to_float<double>()) > 1e-3) {
                pass = false;
                std::cout << "C(i,j,k).to_float<double>() = " << C(i,j,k).to_float<double>() << std::endl;
                std::cout << "expected_C(i,j,k).to_float<double>() = " << expected_C(i,j,k).to_float<double>() << std::endl;
                std::cout << "C(i,j,k).to_float<double>() - expected_C(i,j,k).to_float<double>() = " << C(i,j,k).to_float<double>() - expected_C(i,j,k).to_float<double>() << std::endl;
                break;
            }
        }
        if(pass) {
            std::cout << "Party " << mpc.party << " 3Dx2D MatMul test passed!" << std::endl;
        }
        else {
            std::cout << "Party " << mpc.party << " 3Dx2D MatMul test failed!" << std::endl;
        }
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

    ZeroExtendRandomness<uint64_t, BW, M_BITS, F, K, 2> randomness;
    randomness = read_zero_extend_randomness<uint64_t, BW, M_BITS, F, K, 2>(mpc, 0, 20, 20);

    auto X_ext_share = zero_extend_tensor<uint64_t, BW, M_BITS, F, K, 2, Eigen::RowMajor>(X_m_share, randomness);
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

    ZeroExtendRandomness<uint64_t, BW, M_BITS, F, K, 3> randomness;
    randomness = read_zero_extend_randomness<uint64_t, BW, M_BITS, F, K, 3>(mpc, 2, 2, 2);

    auto X_ext_share = zero_extend_tensor<uint64_t, BW, M_BITS, F, K, 3, Eigen::RowMajor>(X_m_share, randomness);
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
    FixTensorN r_xy_share(D1,D2), r_x_rymsb_share(D1,D2), r_xmsb_y_share(D1,D2);

    ElementwiseMulRandomness<uint64_t, BW, M_BITS, F, K, 2> randomness;
    randomness = read_elementwise_mul_randomness<uint64_t, BW, M_BITS, F, K, 2>(mpc, 0, D1, D2);
    std::cout << "finish reading randomness" << std::endl;
    // 4. Secure computation
    FixTensorN c_n_share = elementwise_mul_opt<uint64_t, BW, M_BITS, F, K, 2, Eigen::RowMajor>(a_m_share, b_m_share, 
        randomness);

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
        std::cout << "test elementwise mul" << std::endl;
        for (int i = 0; i < D1; ++i) {
            for (int j = 0; j < D2; ++j) {
                std::cout << C_trunc(i,j).to_float<double>() << " " << expected_C(i,j).to_float<double>() << std::endl;                
                if (std::abs(C_trunc(i,j).template to_float<double>() - expected_C(i,j).template to_float<double>()) > 1e-1) {
                    std::cout << "C_trunc(i,j).to_float<double>() - expected_C(i,j).template to_float<double>() = " << C_trunc(i,j).val - expected_C(i,j).val << std::endl;
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

void test_square_tensor_opt(MPC& mpc) {
    std::cout << "\n--- Testing Square Tensor Opt for Party " << mpc.party << "/" << mpc.M << " ---" << std::endl;
    mpc.reset_stats();
    if (mpc.M != 2) return;

    constexpr int M_BITS = BW - F;
    using FixTensorM = FixTensor<uint64_t, M_BITS, F, K, 2>;
    using FixTensorBW = FixTensor<uint64_t, BW, F, K, 2>;

    FixTensorM X_m_plain(3,4);
    if (mpc.party == 0) {
        X_m_plain.initialize(0, 16);
    }
    
    FixTensorM X_m_share = secret_share_tensor(X_m_plain);
    
    SquareRandomness<uint64_t, BW, M_BITS, F, K, 2> randomness;
    randomness = read_square_randomness<uint64_t, BW, M_BITS, F, K, 2>(mpc, 0, 3, 4);   
    
    auto X_square_share = square_tensor_opt<uint64_t, BW, M_BITS, F, K, 2, Eigen::RowMajor>(X_m_share, randomness);
    auto X_reconstructed = reconstruct_tensor(X_square_share);

    FixTensorBW X_n_plain = change_bitwidth<BW, F, K>(X_m_plain);
    FixTensorBW X_square_plain = X_n_plain * X_n_plain;

    if (mpc.party == 0) {
        for(int i=0; i<3; ++i) for(int j=0; j<4; ++j) {
            std::cout << X_square_plain(i,j).to_float<double>() << " " << X_reconstructed(i,j).to_float<double>() << std::endl;
        }
        std::cout << "Party " << mpc.party << " Square Tensor Opt test passed!" << std::endl;
    }
    mpc.print_stats();
}

void test_square_tensor_opt_3d(MPC& mpc) {
    std::cout << "\n--- Testing Square Tensor Opt 3D for Party " << mpc.party << "/" << mpc.M << " ---" << std::endl;
    mpc.reset_stats();
    if (mpc.M != 2) return;

    constexpr int M_BITS = BW - F;
    using FixTensorM = FixTensor<uint64_t, M_BITS, F, K, 3>;
    using FixTensorBW = FixTensor<uint64_t, BW, F, K, 3>;

    FixTensorM X_m_plain(20,20,20);
    if (mpc.party == 0) {
        X_m_plain.initialize(0, 16);
    }
    
    FixTensorM X_m_share = secret_share_tensor(X_m_plain);
    
    SquareRandomness<uint64_t, BW, M_BITS, F, K, 3> randomness;
    randomness = read_square_randomness<uint64_t, BW, M_BITS, F, K, 3>(mpc, 20, 20, 20);

    auto X_square_share = square_tensor_opt<uint64_t, BW, M_BITS, F, K, 3>(X_m_share, randomness);
    auto X_reconstructed = reconstruct_tensor(X_square_share);

    FixTensorBW X_n_plain = change_bitwidth<BW, F, K>(X_m_plain);
    FixTensorBW X_square_plain = X_n_plain * X_n_plain;
    auto X_square_trunc = truncate_reduce_tensor(X_square_plain);
    auto X_reconstructed_trunc = truncate_reduce_tensor(X_reconstructed);

    if (mpc.party == 0) {
        for(int i=0; i<2; ++i) for(int j=0; j<2; ++j) for(int k=0; k<2; ++k) {
            std::cout << "X_n_plain(" << i << "," << j << "," << k << ").to_float<double>() = " << X_n_plain(i,j,k).to_float<double>() << std::endl;
            std::cout << "X_square_plain(" << i << "," << j << "," << k << ").to_float<double>() = " << X_square_trunc(i,j,k).to_float<double>() << std::endl;
            std::cout << "X_reconstructed(" << i << "," << j << "," << k << ").to_float<double>() = " << X_reconstructed_trunc(i,j,k).to_float<double>() << std::endl;
            if (std::abs(X_square_trunc(i,j,k).to_float<double>() - X_reconstructed_trunc(i,j,k).to_float<double>()) > 1e-1) {
                std::cout << "Party " << mpc.party << " Square Tensor Opt 3D test FAILED!" << std::endl;
            } else {
                std::cout << "Party " << mpc.party << " Square Tensor Opt 3D test passed!" << std::endl;
            }
        }
    }
    mpc.print_stats();
}

void test_square_scalar_opt(MPC& mpc) {
    std::cout << "\n--- Testing Square Scalar Opt for Party " << mpc.party << "/" << mpc.M << " ---" << std::endl;
    mpc.reset_stats();
    if (mpc.M != 2) return;

    constexpr int M_BITS = BW - F;
    using FixM = Fix<uint64_t, M_BITS, F, K>;
    using FixBW = Fix<uint64_t, BW, F, K>;

    FixM x_m_plain;
    if (mpc.party == 0) {
        x_m_plain = FixM(-3.14);
    }
    
    FixM x_m_share = secret_share(x_m_plain);

    FixM r_m_share;
    mpc.read_fix_share(r_m_share);
    FixBW r_n_share;
    mpc.read_fix_share(r_n_share);
    FixBW r_square_share;
    mpc.read_fix_share(r_square_share);
    FixBW r_msb_share;
    mpc.read_fix_share(r_msb_share);
    FixBW r_r_msb_share;
    mpc.read_fix_share(r_r_msb_share);

    auto x_square_share = square_scalar_opt<uint64_t, BW, M_BITS, F, K>(x_m_share, r_m_share, r_n_share, r_square_share, r_msb_share, r_r_msb_share);
    auto x_reconstructed = truncate_reduce(reconstruct(x_square_share));
    
    if (mpc.party == 0) {
        std::cout << "Square Scalar Opt: x_reconstructed.to_float<double>() = " << x_reconstructed.to_float<double>() << " " << x_m_plain.template to_float<double>() * x_m_plain.template to_float<double>() << std::endl;
        if (std::abs(x_reconstructed.to_float<double>() - x_m_plain.template to_float<double>() * x_m_plain.template to_float<double>()) > 1e-1) {
            std::cout << "Party " << mpc.party << " Square Scalar Opt test FAILED!" << std::endl;
        } else {
            std::cout << "Party " << mpc.party << " Square Scalar Opt test passed!" << std::endl;
        }
    }
    mpc.print_stats();
}

void test_exp_scalar_opt(MPC& mpc) {
    std::cout << "\n--- Testing Exp Scalar Opt for Party " << mpc.party << "/" << mpc.M << " ---" << std::endl;
    mpc.reset_stats();
    if (mpc.M != 2) return;

    constexpr int M_BITS = BW - F;
    using FixM = Fix<uint64_t, M_BITS, F, K>;
    using FixBW = Fix<uint64_t, BW, F, K>;

    FixTensor<uint64_t, M_BITS, F, K, 1> R(RECIPROCAL_NR_ITERS);
    FixTensor<uint64_t, BW, F, K, 1> R_N(RECIPROCAL_NR_ITERS);
    FixTensor<uint64_t, BW, F, K, 1> R_SQUARE(RECIPROCAL_NR_ITERS);
    FixTensor<uint64_t, BW, F, K, 1> R_MSB(RECIPROCAL_NR_ITERS);
    FixTensor<uint64_t, BW, F, K, 1> R_R_MSB(RECIPROCAL_NR_ITERS);
    Fix<uint64_t, M_BITS, F, K> R_M_EXT;
    Fix<uint64_t, BW, F, K> R_E_EXT;
    Fix<uint64_t, BW, F, K> R_MSB_EXT;

    for (int i = 0; i < RECIPROCAL_NR_ITERS; ++i) {
        mpc.read_fix_share(R(i));
        mpc.read_fix_share(R_N(i));
        mpc.read_fix_share(R_SQUARE(i));
        mpc.read_fix_share(R_MSB(i));
        mpc.read_fix_share(R_R_MSB(i));
    }
    mpc.read_fix_share(R_M_EXT);
    mpc.read_fix_share(R_E_EXT);
    mpc.read_fix_share(R_MSB_EXT);

    FixBW x_secret(0);
    if (mpc.party == 0) {
        x_secret = FixBW(-1.14159);
    }
    
    FixBW x_n_share = secret_share(x_secret);
    FixBW x_exp_share = exp_scalar<uint64_t, BW, M_BITS, F, K>(x_n_share, R, R_N, R_SQUARE, R_MSB, R_R_MSB, R_M_EXT, R_E_EXT, R_MSB_EXT);
    FixBW x_exp_reconstructed = reconstruct(x_exp_share);

    if (mpc.party == 0) {
        std::cout << "x_exp_reconstructed.to_float<double>() = " << x_exp_reconstructed.to_float<double>() << " " << std::exp(-1.14159) << std::endl;
        if (std::abs(x_exp_reconstructed.to_float<double>() - std::exp(-1.14159)) > 1e-1) {
            std::cout << "Party " << mpc.party << " Exp Scalar Opt test FAILED!" << std::endl;
        } else {
            std::cout << "Party " << mpc.party << " Exp Scalar Opt test passed!" << std::endl;
        }
    }
    mpc.print_stats();
}

void test_exp_tensor_opt_3d(MPC& mpc) {
    std::cout << "\n--- Testing Exp Tensor Opt 3D for Party " << mpc.party << "/" << mpc.M << " ---" << std::endl;
    mpc.reset_stats();
    if (mpc.M != 2) return;

    constexpr int M_BITS = BW - F;
    using FixTensorM = FixTensor<uint64_t, M_BITS, F, K, 3>;
    using FixTensorBW = FixTensor<uint64_t, BW, F, K, 3>;
    FixTensorBW X_plain(20,20,20);
    Random rg;
    for (int i = 0; i < 20; i++){
        for (int j = 0; j < 20; j++){
            for (int k = 0; k < 20; k++){
                auto val = rg.template randomGE<uint64_t>(1, F)[0];
                X_plain(i,j,k) = Fix<uint64_t, BW, F, K>(double(-val/pow(2.0, F)));
            }
        }
    }
    for(int i = 0; i < 2; ++i) for(int j = 0; j < 2; ++j) for(int k = 0; k < 2; ++k) {
        std::cout << "X_plain(" << i << "," << j << "," << k << ").to_float<double>() = " << X_plain(i,j,k).to_float<double>() << std::endl;
    }
    int iters = RECIPROCAL_NR_ITERS;
    FixTensorBW X_share = secret_share_tensor(X_plain);
    
    ExpRandomness<uint64_t, BW, M_BITS, F, K, 3> randomness;
    randomness = read_exp_randomness<uint64_t, BW, M_BITS, F, K, 3>(mpc, 20, 20, 20);
    auto X_exp_share = exp_tensor<uint64_t, BW, M_BITS, F, K, 3, Eigen::RowMajor>(X_share, randomness);
    auto X_exp_reconstructed = reconstruct_tensor(X_exp_share);

    if (mpc.party == 0) {
        for(int i=0; i<2; ++i) for(int j=0; j<2; ++j) for(int k=0; k<2; ++k) {
            std::cout << "X_plain(" << i << "," << j << "," << k << ").to_float<double>() = " << X_plain(i,j,k).to_float<double>() << std::endl;
            std::cout << "X_exp_reconstructed(" << i << "," << j << "," << k << ").to_float<double>() = " << X_exp_reconstructed(i,j,k).to_float<double>() << std::endl;
        }
        for(int i=0; i<20; ++i) for(int j=0; j<20; ++j) for(int k=0; k<20; ++k) {
            if (std::abs(std::exp(X_plain(i,j,k).to_float<double>()) - X_exp_reconstructed(i,j,k).to_float<double>()) > 1e-1) {
                std::cout << "Party " << mpc.party << " Exp Tensor Opt 3D test FAILED!" << std::endl;
                std::cout << "X_plain(" << i << "," << j << "," << k << ").to_float<double>() = " << X_plain(i,j,k).to_float<double>() << std::endl;
                std::cout << "X_exp_reconstructed(" << i << "," << j << "," << k << ").to_float<double>() = " << X_exp_reconstructed(i,j,k).to_float<double>() << std::endl;
            }
        }
        std::cout << "Party " << mpc.party << " Exp Tensor Opt 3D test passed!" << std::endl;
    }
    mpc.print_stats();
}

void test_inv_sqrt_tensor(MPC& mpc) {
    std::cout << "\n--- Testing Inv Sqrt Tensor for Party " << mpc.party << "/" << mpc.M << " ---" << std::endl;
    mpc.reset_stats();
    if (mpc.M != 2) return;

    constexpr int M_BITS = BW - F;
    using FixTensorM = FixTensor<uint64_t, M_BITS, F, K, 3>;
    using FixTensorBW = FixTensor<uint64_t, BW, F, K, 3>;

    FixTensorBW X_plain(1, 3, 3);
    for (int i = 0; i < 3; i++){
        for (int j = 0; j < 3; j++){
            X_plain(0, i, j) = Fix<uint64_t, BW, F, K>(double(i + j + 1.1));
        }
    }
    FixTensorBW X_n_share = secret_share_tensor(X_plain);
    InvSqrtRandomness<uint64_t, BW, M_BITS, F, K, 3> randomness;
    randomness = read_inv_sqrt_randomness<uint64_t, BW, M_BITS, F, K, 3>(mpc, 1, 3, 3);
    auto X_inv_sqrt_share = inv_sqrt_tensor<uint64_t, BW, M_BITS, F, K, 3>(X_n_share, randomness);
    auto X_inv_sqrt_reconstructed = reconstruct_tensor(X_inv_sqrt_share);
    if (mpc.party == 0) {
        for(int i = 0; i < 3; i++){
            for(int j = 0; j < 3; j++){
                std::cout << "X_plain(0, " << i << ", " << j << ").to_float<double>() = " << X_plain(0, i, j).to_float<double>() << std::endl;
                std::cout << "X_inv_sqrt_reconstructed(0, " << i << ", " << j << ").to_float<double>() = " << X_inv_sqrt_reconstructed(0, i, j).to_float<double>() << std::endl;
            }
        }
        std::cout << "Party " << mpc.party << " Inv Sqrt Tensor test passed!" << std::endl;
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
        // test_tensor_share_and_reconstruct(mpc);
        // test_reconstruct_parallel(mpc);
        test_secure_matmul(mpc);
        test_secure_matmul_3d(mpc);
        // for(int i = 0; i < 10; i++){
        //     test_truncate_zero_extend_scalar(mpc);
        // }
        test_truncate_zero_extend_tensor_2d(mpc);
        test_truncate_zero_extend_tensor_3d(mpc);
        test_elementwise_mul_opt(mpc); 
        test_square_tensor_opt(mpc);
        test_square_tensor_opt_3d(mpc);
        // for(int i = 0; i < 10; i++){
        //     test_square_scalar_opt(mpc);
        // }
        // test_exp_scalar_opt(mpc);
        test_exp_tensor_opt_3d(mpc);
        test_inv_sqrt_tensor(mpc);
        mpc.close();

    } catch (const std::exception& e) {
        std::cerr << "Party " << party << " caught an exception: " << e.what() << std::endl;
        return 1;
    }
    return 0;
}
