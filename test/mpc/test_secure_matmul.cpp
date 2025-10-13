#include "mpc/mpc.h"
#include <iostream>

void test_secure_matmul_external_recon(MPC& mpc) {
    std::cout << "--- Testing secure_matmul with external reconstruction for Party " << mpc.party << "/" << mpc.M << " ---" << std::endl;

    using T = uint64_t;
    const int F = 16;
    const int K_INT = 15;
    const int BW = 64;

    using Fix = Fix<T, BW, F, K_INT>;
    using Tensor2D = FixTensor<T, BW, F, K_INT, 2>;

    // Matrix dimensions
    const int M = 2;
    const int N = 3;
    const int P = 4;

    // 1. Plaintext matrices A and B (only party 0 needs the actual values)
    Tensor2D A_plain(M, N);
    Tensor2D B_plain(N, P);
    if (mpc.party == 0) {
        A_plain.setValues({
            {Fix(1.5), Fix(-2.0), Fix(3.0)},
            {Fix(0.5), Fix(1.0), Fix(-1.0)}
        });
        B_plain.setValues({
            {Fix(1.0), Fix(2.0), Fix(3.0), Fix(4.0)},
            {Fix(-1.0), Fix(0.5), Fix(-2.5), Fix(1.5)},
            {Fix(2.0), Fix(-1.0), Fix(0.0), Fix(1.0)}
        });
    } else {
        A_plain.setZero();
        B_plain.setZero();
    }

    // 2. Secret share A and B
    auto A_share = secret_share_tensor(A_plain);
    auto B_share = secret_share_tensor(B_plain);

    // 3. Create Beaver triple shares (U, V, Z)
    Tensor2D U_share(M, N);
    Tensor2D V_share(N, P);
    Tensor2D Z_share(M, P);
    
    // In a real scenario, these would come from a trusted dealer.
    // Here we just use the randomness loaded from files.
    mpc.read_fixtensor_share(U_share);
    mpc.read_fixtensor_share(V_share);
    mpc.read_fixtensor_share(Z_share);

    // 4. Compute E_share and F_share, then reconstruct them externally
    auto E_share = A_share - U_share;
    auto F_share = B_share - V_share;
    
    auto E_recon = reconstruct_tensor(E_share);
    auto F_recon = reconstruct_tensor(F_share);

    // 5. Call secure_matmul with externally reconstructed E and F
    auto C_share = secure_matmul(A_share, B_share, U_share, V_share, Z_share, &E_recon, &F_recon);

    // 6. Reconstruct the result
    auto C_reconstructed_full = reconstruct_tensor(C_share);

    // 7. Plaintext verification (only party 0)
    if (mpc.party == 0) {
        auto C_expected_full = tensor_mul(A_plain, B_plain);
        auto C_expected = truncate_reduce_tensor(C_expected_full);
        auto C_reconstructed = truncate_reduce_tensor(C_reconstructed_full);

        std::cout << "Reconstructed Output:\n" << C_reconstructed << std::endl;
        std::cout << "Expected Output:\n" << C_expected << std::endl;

        for (int i = 0; i < C_reconstructed.size(); ++i) {
            // Allow a small tolerance due to fixed-point arithmetic
            assert(std::abs(static_cast<int64_t>(C_reconstructed.data()[i].val) - static_cast<int64_t>(C_expected.data()[i].val)) <= 1);
        }
        std::cout << "Verification successful!" << std::endl;
    }
    
    std::cout << "Party " << mpc.party << " test passed!" << std::endl;
}

int main(int argc, char** argv) {
    if (argc < 2) {
        std::cout << "Usage: " << argv[0] << " <party>" << std::endl;
        return 1;
    }
    int party = std::atoi(argv[1]);
    MPC mpc(2, party);
    
    // We reuse the randomness file from the FC test for convenience
    std::string randomness_path = "randomness/P" + std::to_string(party) + "/matmul_random_data.bin";
    mpc.load_random_data(randomness_path);
    
    std::vector<std::string> addrs = {"127.0.0.1", "127.0.0.1"};
    mpc.connect(addrs, 9005); // Using a different port to avoid conflicts

    test_secure_matmul_external_recon(mpc);

    return 0;
}
