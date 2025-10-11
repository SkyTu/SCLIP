#ifndef SCLIP_MPC_H
#define SCLIP_MPC_H

#include "utils/comm.h"
#include "mpc/fix.h"
#include "mpc/fix_tensor.h"
#include "mpc/tensor_ops.h"
#include "utils/random.h"
#include "utils/compress.h"
#include <vector>
#include <string>
#include <stdexcept>
#include <algorithm>
#include <thread>
#include <mutex>
#include <fstream>
#include <future>
#include <type_traits>

// Forward declaration for MPC class to be used in friend declaration
class MPC; 

// A global MPC instance to be accessible by standalone protocol functions
extern MPC* mpc_instance;

class MPC {
public:
    int M; // Number of parties
    int party;
    std::vector<TCPPeer> peers;
    std::vector<uint8_t> random_data;
    size_t random_data_idx;

    MPC(int m, int p) : M(m), party(p), random_data_idx(0) {
        if (p >= m) {
            throw std::runtime_error("Party number must be less than M.");
        }
        peers.resize(m);
        mpc_instance = this;
    }

    ~MPC() {
        mpc_instance = nullptr;
    }

    void connect(const std::vector<std::string>& addrs, int port_offset = 8000) {
        if (addrs.size() != M) {
            throw std::runtime_error("Number of addresses must be equal to M.");
        }

        std::vector<std::thread> threads;
        for (int i = 0; i < M; ++i) {
            if (i == party) continue;

            threads.emplace_back([this, i, &addrs, port_offset]() {
                bool is_server = (party < i);
                std::string target_addr = is_server ? "" : addrs[i];
                
                int p_min = std::min(party, i);
                int p_max = std::max(party, i);
                int port = port_offset + p_min * M + p_max;
                
                {
                    std::lock_guard<std::mutex> lock(cout_mutex);
                    if (is_server) {
                        std::cout << "[Party " << party << "] Listening for Party " << i << " on port " << port << "..." << std::endl;
                    } else {
                        std::cout << "[Party " << party << "] Connecting to Party " << i << " on port " << port << "..." << std::endl;
                    }
                }

                peers[i].connect(target_addr, port, is_server);
                
                {
                    std::lock_guard<std::mutex> lock(cout_mutex);
                    std::cout << "[Party " << party << "] <-> [Party " << i << "] connection established." << std::endl;
                }
            });
        }
        // Wait for all connection threads to complete.
        for (auto& t : threads) {
            t.join();
        }
    }

    void load_random_data(const std::string& filename) {
        std::ifstream in(filename, std::ios::binary | std::ios::ate);
        if (!in) {
            throw std::runtime_error("Cannot open random data file: " + filename);
        }
        if (!in.is_open()) {
            throw std::runtime_error("Could not open random data file: " + filename);
        }
        in.seekg(0, std::ios::end);
        std::streamsize size = in.tellg();
        if (size == -1) {
            throw std::runtime_error("Failed to get size of random data file: " + filename);
        }
        in.seekg(0, std::ios::beg);

        random_data.resize(size);
        if (in.read(reinterpret_cast<char*>(random_data.data()), size)) {
            // Successfully read file
        } else {
            throw std::runtime_error("Failed to read random data from file: " + filename);
        }

        in.close();
        // Reset random data consumption index when loading a new random data file
        random_data_idx = 0;
    }

    Random& get_random_generator() { return random_gen; }

public:
    // Helper to get random uint64_t share from the buffer
    uint64_t get_next_uint64_share(int bitwidth) {
        if (random_data_idx + ((bitwidth + 7) / 8) > random_data.size()) {
            throw std::runtime_error("Not enough random data in buffer for uint64_t share.");
        }
        size_t bytes_needed = (bitwidth + 7) / 8;
        std::vector<uint8_t> packed_data(random_data.begin() + random_data_idx, random_data.begin() + random_data_idx + bytes_needed);
        std::vector<uint64_t> unpacked = unpack_data<uint64_t>(packed_data, 1, bitwidth);
        random_data_idx += bytes_needed;
        return unpacked[0];
    }

    // Helper to get random Fix share from the buffer
    template <typename T, int bw, int f, int k>
    Fix<T, bw, f, k> get_next_fix_share() {
        return Fix<T, bw, f, k>(get_next_uint64_share(bw));
    }

    // Helper to read a raw Fix share from the buffer
    template <typename FixType>
    void read_fix_share(FixType& scalar) {
        size_t bytes_needed = sizeof(typename FixType::val_type);
        if (random_data_idx + bytes_needed > random_data.size()) {
            throw std::runtime_error("Not enough random data for fix share.");
        }
        memcpy(&scalar.val, random_data.data() + random_data_idx, bytes_needed);
        random_data_idx += bytes_needed;
    }

    // New, safer function to read a tensor share into a pre-sized tensor
    template <typename FixTensorType>
    void read_fixtensor_share(FixTensorType& tensor) {
        long long num_elements = tensor.size();
        if (num_elements == 0) return; // Nothing to read

        // Our dealer writes raw Fix<T> objects, not packed bits.
        size_t bytes_needed = num_elements * sizeof(typename FixTensorType::Scalar);

        if (random_data_idx + bytes_needed > random_data.size()) {
            throw std::runtime_error("Not enough random data in buffer for tensor data.");
        }
        
        memcpy(tensor.data(), random_data.data() + random_data_idx, bytes_needed);
        random_data_idx += bytes_needed;
    }

public:
    void close() {
        // TCPPeer destructor will handle closing sockets
        peers.clear();
    }

    void reset_stats() {
        for (int i = 0; i < M; ++i) {
            if (i == party) continue;
            peers[i].reset_stats();
        }
    }

    void print_stats() {
        size_t total_sent = 0;
        size_t total_received = 0;
        for (int i = 0; i < M; ++i) {
            if (i == party) continue;
            total_sent += peers[i].get_bytes_sent();
            total_received += peers[i].get_bytes_received();
        }
        std::cout << "Party " << party << " Stats --- Sent: " << total_sent 
                  << " bytes, Received: " << total_received 
                  << " bytes" << std::endl;
    }
private:
    Random random_gen;
    std::mutex cout_mutex;
    std::vector<std::thread> threads;
};

// ================= Protocol Implementations =================

template <typename T, int bw, int f, int k>
void send_compressed(int party_to, const Fix<T, bw, f, k>& x) {
    uint64_t val = x.val;
    std::vector<uint8_t> packed = pack_data(&val, 1, bw);
    mpc_instance->peers[party_to].send_data(packed.data(), packed.size());
}

template <typename T, int bw, int f, int k>
Fix<T, bw, f, k> recv_compressed(int party_from) {
    size_t bytes_to_recv = (bw + 7) / 8;
    std::vector<uint8_t> packed(bytes_to_recv);
    mpc_instance->peers[party_from].recv_data(packed.data(), packed.size());
    std::vector<T> unpacked = unpack_data<T>(packed, 1, bw);
    return Fix<T, bw, f, k>(unpacked[0]);
}


template <typename T, int bw, int f, int k>
Fix<T, bw, f, k> secret_share(Fix<T, bw, f, k> x) {
    if (mpc_instance == nullptr) throw std::runtime_error("MPC instance not initialized.");
    int party = mpc_instance->party;
    int M = mpc_instance->M;

    if (party == 0) {
        T* rand_vals_ptr = mpc_instance->get_random_generator().template randomGE<T>(M - 1, bw);
        std::vector<Fix<T, bw, f, k>> shares;
        Fix<T, bw, f, k> sum_share(0);

        for (int i = 0; i < M - 1; ++i) {
            Fix<T, bw, f, k> share(rand_vals_ptr[i]);
            shares.push_back(share);
            sum_share = sum_share + share;
        }

        Fix<T, bw, f, k> my_share = x - sum_share;

        for (int i = 0; i < M - 1; ++i) {
            send_compressed<T, bw, f, k>(i + 1, shares[i]);
        }
        delete[] rand_vals_ptr;
        return my_share;
    } else {
        return recv_compressed<T, bw, f, k>(0);
    }
}


template <typename T, int bw, int f, int k>
Fix<T, bw, f,k> reconstruct(Fix<T, bw, f,k> my_share) {
    if (mpc_instance == nullptr) {
        throw std::runtime_error("MPC instance is not initialized.");
    }
    int M = mpc_instance->M;
    int party = mpc_instance->party;


    // Stage 1: All parties send their share to all other parties.
    for (int i = 0; i < M; ++i) {
        if (i == party) continue;
        send_compressed<T, bw, f, k>(i, my_share);
    }

    // Stage 2: All parties receive shares from all other parties.
    std::vector<Fix<T, bw, f, k>> all_shares(M);
    all_shares[party] = my_share;
    for (int i = 0; i < M; ++i) {
        if (i == party) continue;
        all_shares[i] = recv_compressed<T, bw, f, k>(i);
    }
    
    Fix<T, bw, f, k> total; // Inits to 0
    for(int i = 0; i < M; ++i) {
        total += all_shares[i];
    }
    return total;
}


// Overload: secure_mul with provided Beaver triple shares (u,v,z)
template <typename T, int bw, int f, int k>
Fix<T, bw, f, k> secure_mul(Fix<T, bw, f, k> x_share,
                             Fix<T, bw, f, k> y_share,
                             Fix<T, bw, f, k> u_share,
                             Fix<T, bw, f, k> v_share,
                             Fix<T, bw, f, k> z_share) {
    if (mpc_instance == nullptr) throw std::runtime_error("MPC instance not initialized.");
    if (mpc_instance->M != 2) throw std::runtime_error("This secure_mul function only supports 2 parties.");

    int party = mpc_instance->party;

    Fix<T, bw, f, k> e_share = x_share - u_share;
    Fix<T, bw, f, k> f_share = y_share - v_share;

    Fix<T, bw, f, k> e = reconstruct(e_share);
    Fix<T, bw, f, k> f_rec = reconstruct(f_share);

    Fix<T, bw, f, k> output_share = e * y_share + f_rec * x_share + z_share;
    if (party == 1) {
        output_share -= e * f_rec;
    }
    return output_share;
}
// ================= Tensor Protocol Implementations =================

template <typename T, int bw, int f, int k, int Rank>
void send_compressed_tensor(int party_to, const FixTensor<T, bw, f, k, Rank>& tensor) {
    // 1. Send dimensions
    mpc_instance->peers[party_to].send_data(tensor.dimensions().data(), Rank * sizeof(long));

    // 2. Send data
    long long num_elements = tensor.size();
    // The underlying data type of Fix is T, so we can cast it.
    const T* raw_data = reinterpret_cast<const T*>(tensor.data());
    std::vector<uint8_t> packed = pack_data(raw_data, num_elements, bw);
    mpc_instance->peers[party_to].send_data(packed.data(), packed.size());
}

template <typename FixTensorType>
FixTensorType recv_compressed_tensor(int party_from) {
    using T = typename FixTensorType::FixType::val_type;
    constexpr int Rank = FixTensorType::Base::NumIndices;
    constexpr int bw = FixTensorType::FixType::bitwidth;
    constexpr int f = FixTensorType::FixType::frac_bits;
    constexpr int k = FixTensorType::FixType::int_bits;

    // 1. Recv dimensions
    Eigen::array<long, Rank> dims;
    mpc_instance->peers[party_from].recv_data(dims.data(), Rank * sizeof(long));

    // 2. Recv data
    FixTensorType result(dims);
    long long num_elements = result.size();
    if (num_elements == 0) {
        return result;
    }
    size_t bytes_to_recv = (num_elements * bw + 7) / 8;
    std::vector<uint8_t> packed(bytes_to_recv);
    mpc_instance->peers[party_from].recv_data(packed.data(), packed.size());
    
    std::vector<T> unpacked = unpack_data<T>(packed, num_elements, bw);

    // The memory layout of vector<T> and FixTensor of Fix<T,...> is compatible.
    memcpy(result.data(), unpacked.data(), num_elements * sizeof(Fix<T, bw, f, k>));

    return result;
}

// ================= Truncate Reduce (logical right shift by f, shrink bitwidth to m=bw-f) =================

template <typename T, int bw, int f, int k>
Fix<T, (bw - f), f, k> truncate_reduce(const Fix<T, bw, f, k>& x_share) {
    static_assert(bw > f, "truncate_reduce requires bw > f");
    constexpr int m = bw - f;
    T new_val = static_cast<T>(x_share.val >> f);
    return Fix<T, m, f, k>(new_val);
}

template <typename T, int bw, int f, int k, int Rank, int Options>
FixTensor<T, (bw - f), f, k, Rank, Options>
truncate_reduce_tensor(const FixTensor<T, bw, f, k, Rank, Options>& x_share) {
    static_assert(bw > f, "truncate_reduce_tensor requires bw > f");
    constexpr int m = bw - f;
    FixTensor<T, m, f, k, Rank, Options> result(x_share.dimensions());
    for (long long i = 0; i < x_share.size(); ++i) {
        result.data()[i] = Fix<T, m, f, k>(static_cast<T>(x_share.data()[i].val >> f));
    }
    return result;
}

// ================= Zero Extend using extension triples (input m-bit ring -> output bw > m) =================

// Layout per element (per party): [r_m_share (m-ring), r_e_share (bw-ring), r_msb_share (bw-ring)]

template <typename T, int m, int f, int k, int bw>
Fix<T, bw, f, k> zero_extend(const Fix<T, m, f, k>& x_m_share, const Fix<T, m, f, k>& r_m_share, const Fix<T, bw, f, k>& r_e_share, const Fix<T, bw, f, k>& r_msb_share) {
    static_assert(bw > m, "zero_extend requires bw > m");
    if (mpc_instance == nullptr) throw std::runtime_error("MPC instance not initialized.");
    
    const T bw_mask = (bw == 64) ? ~T(0) : ((T(1) << bw) - 1);

    int M = mpc_instance->M;
    int party = mpc_instance->party;
    Fix<T, m, f, k> bias = (party == 1 && m >= 2) ? Fix<T, m, f, k>(static_cast<T>(T(1) << (m - 2))) : Fix<T, m, f, k>(static_cast<T>(0));
    // 1) xhat_m_share = (x_m_share + r_m_share) in m-ring
    Fix<T, m, f, k> xhat_share_m = x_m_share + r_m_share + bias;

    // 2) Reconstruct xhat (public), then MSB at bit (m-1)
    Fix<T, m, f, k> xhat_m_public = reconstruct(xhat_share_m);
    T msb = (xhat_m_public.val >> (m - 1)) & T(1);

    // 3) e = 2^m * <r^msb>  (in bw-ring)
    T two_pow_m = (m >= 64) ? T(0) : (T(1) << m);
    T e_share_val = (two_pow_m == T(0)) ? T(0) : (r_msb_share.val * two_pow_m) & bw_mask;

    // 4) t = <e> * (1 - MSB(xhat))
    T one_minus_msb = T(1) - msb;
    T t_share_val = (e_share_val * one_minus_msb) & bw_mask;

    // 5) Return σ·(x̂ − 2^{m−2}) − <r^e> + <t>  (mod 2^bw)
    T bias_val = (m >= 2) ? (T(1) << (m - 2)) : T(0);
    T term_sigma = (party == 1 ? ((xhat_m_public.val + bw_mask + 1) - bias_val) & bw_mask : T(0));
    T xext_val = (term_sigma + ((t_share_val + bw_mask + 1) - r_e_share.val)) & bw_mask;
    return Fix<T, bw, f, k>(xext_val);
}

// when the input is a reconstructed x_hat, not a share
template <typename T, int m, int f, int k, int bw>
Fix<T, bw, f, k> zero_extend_reconstructed(const Fix<T, m, f, k>& x_hat, const Fix<T, m, f, k>& r_m_share, const Fix<T, bw, f, k>& r_e_share, const Fix<T, bw, f, k>& r_msb_share) {
    static_assert(bw > m, "zero_extend requires bw > m");
    if (mpc_instance == nullptr) throw std::runtime_error("MPC instance not initialized.");
    int party = mpc_instance->party;
    const T bw_mask = (bw == 64) ? ~T(0) : ((T(1) << bw) - 1);
    Fix<T, m, f, k> bias = (m >= 2) ? Fix<T, m, f, k>(static_cast<T>(T(1) << (m - 2))) : Fix<T, m, f, k>(static_cast<T>(0));
    // 1) xhat_m_share = (x_m_share + r_m_share) in m-ring
    Fix<T, m, f, k> x_hat_m_public = x_hat + bias;

    T msb = (x_hat_m_public.val >> (m - 1)) & T(1);

    // 3) e = 2^m * <r^msb>  (in bw-ring)
    T two_pow_m = (m >= 64) ? T(0) : (T(1) << m);
    T e_share_val = (two_pow_m == T(0)) ? T(0) : (r_msb_share.val * two_pow_m) & bw_mask;

    // 4) t = <e> * (1 - MSB(xhat))
    T one_minus_msb = T(1) - msb;
    T t_share_val = (e_share_val * one_minus_msb) & bw_mask;

    // 5) Return σ·(x̂ − 2^{m−2}) − <r^e> + <t>  (mod 2^bw)
    T bias_val = (m >= 2) ? (T(1) << (m - 2)) : T(0);
    T term_sigma = (party == 1 ? ((x_hat_m_public.val + bw_mask + 1) - bias_val) & bw_mask : T(0));
    T xext_val = (term_sigma + ((t_share_val + bw_mask + 1) - r_e_share.val)) & bw_mask;
    return Fix<T, bw, f, k>(xext_val);
}

// Overload: zero_extend_tensor with provided r_m, r_e, r_msb shares
template <typename T, int m, int f, int k, int Rank, int Options, int bw>
FixTensor<T, bw, f, k, Rank, Options>
zero_extend_tensor(const FixTensor<T, m, f, k, Rank, Options>& x_m_share,
                   const FixTensor<T, m, f, k, Rank, Options>& r_m_share,
                   const FixTensor<T, bw, f, k, Rank, Options>& r_e_share,
                   const FixTensor<T, bw, f, k, Rank, Options>& r_msb_share) {
    static_assert(bw > m, "zero_extend_tensor requires bw > m");
    if (mpc_instance == nullptr) throw std::runtime_error("MPC instance not initialized.");
 
    int party = mpc_instance->party;
    const T bw_mask = (bw == 64) ? ~ T(0) : ((T(1) << bw) - 1);

    FixTensor<T, m, f, k, Rank, Options> bias(x_m_share.dimensions());
    
    T bias_val = (m >= 2) ? (T(1) << (m - 2)) : T(0);
    
    bias.setConstant(Fix<T, m, f, k>(bias_val));
    auto xhat_m_share = x_m_share + r_m_share;
    FixTensor<T, m, f, k, Rank, Options>xhat_public = reconstruct_tensor(xhat_m_share) + bias;
 
    FixTensor<T, bw, f, k, Rank, Options> result(x_m_share.dimensions());
    T two_pow_m = (m >= 64) ? 0 : (T(1) << m);
 
    for (long long i = 0; i < x_m_share.size(); ++i) {
        T msb = (xhat_public.data()[i].val >> (m - 1)) & T(1);
        T e_share_val = (two_pow_m == T(0)) ? T(0) : (r_msb_share.data()[i].val * two_pow_m) & bw_mask;
        T one_minus_msb = T(1) - msb;
        T t_share_val = (e_share_val * one_minus_msb) & bw_mask;
        
        T term_sigma = (party == 1 ? ((xhat_public.data()[i].val + bw_mask + 1) - bias_val) & bw_mask : T(0));
        T xext_val = (term_sigma + ((t_share_val + bw_mask + 1) - r_e_share.data()[i].val)) & bw_mask;
        
        result.data()[i] = Fix<T, bw, f, k>(xext_val);
    }
 
    return result;
}


template <typename FixTensorType>
FixTensorType secret_share_tensor(const FixTensorType& x) {
    if (mpc_instance == nullptr) throw std::runtime_error("MPC instance not initialized.");
    int party = mpc_instance->party;

    if (party == 0) {
        FixTensorType r(x.dimensions());
        r.initialize();
        send_compressed_tensor(1, r);
        return x - r;
    } else {
        auto r = recv_compressed_tensor<FixTensorType>(0);
        return r;
    }
}

template <typename FixTensorType>
FixTensorType reconstruct_tensor(const FixTensorType& x_share) {
    if (mpc_instance == nullptr) throw std::runtime_error("MPC instance not initialized.");
    int party = mpc_instance->party;
    int M = mpc_instance->M;

    // Stage 1: All parties send their share to all other parties.
    for (int i = 0; i < M; ++i) {
        if (i == party) continue;
        send_compressed_tensor(i, x_share);
    }

    // Stage 2: All parties receive shares from all other parties.
    FixTensorType total = x_share;
    for (int i = 0; i < M; ++i) {
        if (i == party) continue;
        FixTensorType other_share = recv_compressed_tensor<FixTensorType>(i);
        total = total + other_share;
    }
    
    return total;
}


// Overload: secure_matmul with provided tensor Beaver triple shares (U,V,Z)
template<
    typename T, int bw, int f, int k,
    int RankX, int RankY,
    template<typename, int, int, int, int, int> class FixTensorT
>
auto secure_matmul(
    const FixTensorT<T, bw, f, k, RankX, Eigen::RowMajor>& x_share,
    const FixTensorT<T, bw, f, k, RankY, Eigen::RowMajor>& y_share,
    const FixTensorT<T, bw, f, k, RankX, Eigen::RowMajor>& u_share,
    const FixTensorT<T, bw, f, k, RankY, Eigen::RowMajor>& v_share,
    const FixTensorT<T, bw, f, k, (RankX - 1 + RankY - 1), Eigen::RowMajor>& z_share,
    const FixTensorT<T, bw, f, k, RankX, Eigen::RowMajor>* e_ptr = nullptr,
    const FixTensorT<T, bw, f, k, RankY, Eigen::RowMajor>* f_rec_ptr = nullptr
) -> FixTensorT<T, bw, f, k, RankX - 1 + RankY - 1, Eigen::RowMajor>
{
    if (mpc_instance == nullptr) throw std::runtime_error("MPC instance not initialized.");
    if (mpc_instance->M != 2) throw std::runtime_error("This secure_matmul function only supports 2 parties.");

    int party = mpc_instance->party;

    using FixTensorX = FixTensorT<T, bw, f, k, RankX, Eigen::RowMajor>;
    using FixTensorY = FixTensorT<T, bw, f, k, RankY, Eigen::RowMajor>;
    using FixTensorZ = FixTensorT<T, bw, f, k, (RankX - 1 + RankY - 1), Eigen::RowMajor>;

    FixTensorX e_val;
    if (e_ptr == nullptr) {
        FixTensorX e_share = x_share - u_share;
        e_val = reconstruct_tensor(e_share);
    }
    const FixTensorX& e = (e_ptr == nullptr) ? e_val : *e_ptr;

    FixTensorY f_rec_val;
    if (f_rec_ptr == nullptr){
        FixTensorY f_share = y_share - v_share;
        f_rec_val = reconstruct_tensor(f_share);
    }
    const FixTensorY& f_rec = (f_rec_ptr == nullptr) ? f_rec_val : *f_rec_ptr;

    FixTensorZ term1 = tensor_mul(e, y_share);
    FixTensorZ term2 = tensor_mul(x_share, f_rec);
    FixTensorZ output_share = term1 + term2 + z_share;
    if (party == 1) {
        FixTensorZ ef_prod = tensor_mul(e, f_rec);
        output_share = output_share - ef_prod;
    }
    return output_share;
}


// Overload: 3Dx2D with provided triples U,V,Z
template<
    typename T, int bw, int f, int k,
    template<typename, int, int, int, int, int> class FixTensorT
>
auto secure_matmul(
    const FixTensorT<T, bw, f, k, 3, Eigen::RowMajor>& x_share,
    const FixTensorT<T, bw, f, k, 2, Eigen::RowMajor>& y_share,
    const FixTensorT<T, bw, f, k, 3, Eigen::RowMajor>& u_share,
    const FixTensorT<T, bw, f, k, 2, Eigen::RowMajor>& v_share,
    const FixTensorT<T, bw, f, k, 3, Eigen::RowMajor>& z_share,
    const FixTensorT<T, bw, f, k, 3, Eigen::RowMajor>* e_ptr = nullptr,
    const FixTensorT<T, bw, f, k, 2, Eigen::RowMajor>* f_rec_ptr = nullptr
) -> FixTensorT<T, bw, f, k, 3, Eigen::RowMajor>
{
    if (mpc_instance == nullptr) throw std::runtime_error("MPC instance not initialized.");
    if (mpc_instance->M != 2) throw std::runtime_error("This secure_matmul function only supports 2 parties.");

    int party = mpc_instance->party;

    using FixTensorX = FixTensorT<T, bw, f, k, 3, Eigen::RowMajor>;
    using FixTensorY = FixTensorT<T, bw, f, k, 2, Eigen::RowMajor>;
    using FixTensorZ = FixTensorT<T, bw, f, k, 3, Eigen::RowMajor>;

    FixTensorX e_val;
    if (e_ptr == nullptr) {
        FixTensorX e_share = x_share - u_share;
        e_val = reconstruct_tensor(e_share);
    }
    const FixTensorX& e = (e_ptr == nullptr) ? e_val : *e_ptr;

    FixTensorY f_rec_val;
    if (f_rec_ptr == nullptr){
        FixTensorY f_share = y_share - v_share;
        f_rec_val = reconstruct_tensor(f_share);
    }
    const FixTensorY& f_rec = (f_rec_ptr == nullptr) ? f_rec_val : *f_rec_ptr;

    FixTensorZ term1 = tensor_mul(e, y_share);
    FixTensorZ term2 = tensor_mul(x_share, f_rec);
    FixTensorZ output_share = term1 + term2 + z_share;
    if (party == 1) {
        FixTensorZ ef_prod = tensor_mul(e, f_rec);
        output_share = output_share - ef_prod;
    }
    return output_share;
}

// Optimized Element-wise Multiplication Protocol from PDF
template <typename T, int m, int f, int k, int n, int Rank, int Options>
auto elementwise_mul_opt(
    const FixTensor<T, m, f, k, Rank, Options>& x_m_share,
    const FixTensor<T, m, f, k, Rank, Options>& y_m_share,
    const FixTensor<T, m, f, k, Rank, Options>& rx_m_share,
    const FixTensor<T, n, f, k, Rank, Options>& rx_n_share,
    const FixTensor<T, n, f, k, Rank, Options>& rx_msb_n_share,
    const FixTensor<T, m, f, k, Rank, Options>& ry_m_share,
    const FixTensor<T, n, f, k, Rank, Options>& ry_n_share,
    const FixTensor<T, n, f, k, Rank, Options>& ry_msb_n_share,
    const FixTensor<T, n, f, k, Rank, Options>& rxy_n_share,
    const FixTensor<T, n, f, k, Rank, Options>& rx_msby_n_share,
    const FixTensor<T, n, f, k, Rank, Options>& rxy_msb_n_share,
    const FixTensor<T, n, f, k, Rank, Options>& rx_msby_msb_n_share
) -> FixTensor<T, n, f, k, Rank, Options>
{
    if (mpc_instance == nullptr) throw std::runtime_error("MPC instance not initialized.");
    
    auto x_hat = reconstruct_tensor(x_m_share + rx_m_share);
    auto y_hat = reconstruct_tensor(y_m_share + ry_m_share);

    T two_pow_m_minus_2_val = (m < 2 || m - 2 >= 64) ? 0 : (T(1) << (m - 2));
    FixTensor<T, m, f, k, Rank, Options> const_term_m(x_hat.dimensions());
    std::cout << "m " << m << std::endl;
    const_term_m.setConstant(Fix<T,m,f,k>(two_pow_m_minus_2_val));
    FixTensor<T, m, f, k, Rank, Options> x_hat_prime = x_hat + const_term_m;
    FixTensor<T, m, f, k, Rank, Options> y_hat_prime = y_hat + const_term_m;
    

    T two_pow_m_val = (m >= 64) ? 0 : (T(1) << m);
    auto ones_n = FixTensor<T, n, f, k, Rank, Options>(x_hat.dimensions());
    ones_n.setConstant(Fix<T,n,f,k>(1));
    auto two_pow_m_n = FixTensor<T, n, f, k, Rank, Options>(x_hat.dimensions());
    two_pow_m_n.setConstant(Fix<T,n,f,k>(two_pow_m_val));
    
    auto t_x = (ones_n - get_msb<n,f,k>(x_hat_prime)) * two_pow_m_n;
    auto t_y = (ones_n - get_msb<n,f,k>(y_hat_prime)) * two_pow_m_n;

    FixTensor<T, n, f, k, Rank, Options> x_hat_prime_n = extend_locally<n,f,k>(x_hat_prime);
    FixTensor<T, n, f, k, Rank, Options> y_hat_prime_n = extend_locally<n,f,k>(y_hat_prime);
    FixTensor<T, n, f, k, Rank, Options> const_term_n(x_hat.dimensions());
    const_term_n.setConstant(Fix<T,n,f,k>(two_pow_m_minus_2_val));
    x_hat_prime_n = x_hat_prime_n - const_term_n;
    y_hat_prime_n = y_hat_prime_n - const_term_n;
    auto term1 = FixTensor<T, n, f, k, Rank, Options>(x_hat.dimensions());
    if (mpc_instance->party == 0){
        term1.setConstant(Fix<T,n,f,k>(0));
    }
    else{
        term1 = x_hat_prime_n * y_hat_prime_n;
    }
    auto term2 = x_hat_prime_n * ry_n_share;
    auto term3 = x_hat_prime_n * t_y * ry_msb_n_share;
    auto term4 = rx_n_share * y_hat_prime_n;
    auto term5 = rxy_n_share;
    auto term6 = t_y * rxy_msb_n_share;
    auto term7 = t_x * rx_msb_n_share * y_hat_prime_n;
    auto term8 = t_x * rx_msby_n_share;
    auto term9 = t_x * t_y * rx_msby_msb_n_share;
    
    std::cout << "term1 " << term1.data()[0].val << std::endl;
    std::cout << "term2 " << term2.data()[0].val << std::endl;
    std::cout << "term3 " << term3.data()[0].val << std::endl;
    std::cout << "term4 " << term4.data()[0].val << std::endl;
    std::cout << "term5 " << term5.data()[0].val << std::endl;
    std::cout << "term6 " << term6.data()[0].val << std::endl;
    std::cout << "term7 " << term7.data()[0].val << std::endl;
    std::cout << "term8 " << term8.data()[0].val << std::endl;
    std::cout << "term9 " << term9.data()[0].val << std::endl;

    FixTensor<T, n, f, k, Rank, Options> result = term1 - term2 + term3 - term4 + term5 - term6 + term7 - term8 + term9;
    return result;
}

#endif //SCLIP_MPC_H
