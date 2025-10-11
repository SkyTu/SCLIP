#include "mpc/fix.h"
#include "utils/random.h"
#include <cassert>
#include <iostream>
#include <vector>

// Simple unit tests for Fix::extend<extend_bw>() sign/zero extension

int main() {
    // Use small formats to make reasoning easy
    using T = uint64_t;
    constexpr int BW = 16;   // total bits
    constexpr int F  = 4;    // frac bits (kept constant)
    constexpr int K  = 11;   // int bits

    using Fix16 = Fix<T, BW, F, K>;

    // Positive value: 3.5
    Fix16 pos(3.5);
    auto pos_ext = pos.extend<8>(); // -> 24-bit format
    assert(std::abs(pos.to_float<double>() - pos_ext.to_float<double>()) < 1e-9);

    // Negative value: -7.25
    Fix16 neg(-7.25);
    auto neg_ext = neg.extend<8>();
    assert(std::abs(neg.to_float<double>() - neg_ext.to_float<double>()) < 1e-9);

    // Byte IO roundtrip example (independent RNG serialization)
    {
        std::vector<uint64_t> raw = {static_cast<uint64_t>(pos.val), static_cast<uint64_t>(neg.val)};
        auto bytes = uints_to_bytes(raw, BW);
        write_bytes_with_length(bytes, "./dataset/zero_extend_bytes.bin");
        auto loaded = read_bytes_with_length("./dataset/zero_extend_bytes.bin");
        auto back = bytes_to_uints(loaded, BW);
        assert(back.size() == raw.size());
        // Compare low BW bits only
        uint64_t mask = (BW == 64) ? ~uint64_t(0) : ((uint64_t(1) << BW) - 1);
        assert((back[0] & mask) == (raw[0] & mask));
        assert((back[1] & mask) == (raw[1] & mask));
    }

    std::cout << "Fix::extend tests passed (and IO roundtrip): "
              << pos_ext.to_float<double>() << ", "
              << neg_ext.to_float<double>() << std::endl;
    return 0;
}


