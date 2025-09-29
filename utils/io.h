#pragma once

#include <vector>
#include <string>
#include <fstream>
#include <cstdint>
#include <stdexcept>
#include <type_traits>

// Forward declaration is needed for the is_base_of check inside the template.
struct FixBase;

// Bit-packing utilities
template <typename T>
std::vector<uint8_t> uints_to_bytes(const T* data, long num_elements, int bitwidth) {
    if (num_elements == 0) return {};
    size_t total_bits = (size_t)num_elements * bitwidth;
    size_t total_bytes = (total_bits + 7) / 8;
    std::vector<uint8_t> bytes(total_bytes, 0);

    for (long i = 0; i < num_elements; ++i) {
        uint64_t val;
        // Check if T is a Fix type or a plain integer
        if constexpr (std::is_base_of<FixBase, T>::value) {
            val = data[i].val;
        } else {
            val = static_cast<uint64_t>(data[i]);
        }

        for (int j = 0; j < bitwidth; ++j) {
            if ((val >> j) & 1) {
                size_t bit_pos = i * bitwidth + j;
                bytes[bit_pos / 8] |= (1 << (bit_pos % 8));
            }
        }
    }
    return bytes;
}

template <typename T>
std::vector<T> bytes_to_uints(const std::vector<uint8_t>& bytes, long num_elements, int bitwidth) {
    if (num_elements == 0) return {};
    std::vector<T> uints(num_elements);
    
    for (long i = 0; i < num_elements; ++i) {
        uint64_t val = 0;
        for (int j = 0; j < bitwidth; ++j) {
            size_t bit_pos = i * bitwidth + j;
            if (bit_pos / 8 < bytes.size() && ((bytes[bit_pos / 8] >> (bit_pos % 8)) & 1)) {
                val |= (1ULL << j);
            }
        }
        if constexpr (std::is_base_of<FixBase, T>::value) {
            uints[i] = T(val);
        } else {
            uints[i] = static_cast<T>(val);
        }
    }
    return uints;
}


// File I/O utilities
inline void write_bytes_with_length(const std::vector<uint8_t>& bytes, const std::string& filename) {
    std::ofstream out(filename, std::ios::binary);
    uint64_t len = bytes.size();
    out.write(reinterpret_cast<const char*>(&len), sizeof(len));
    out.write(reinterpret_cast<const char*>(bytes.data()), len);
    out.close();
}

inline std::vector<uint8_t> read_bytes_with_length(const std::string& filename) {
    std::ifstream in(filename, std::ios::binary);
    if (!in) {
        throw std::runtime_error("Cannot open file: " + filename);
    }
    uint64_t len;
    in.read(reinterpret_cast<char*>(&len), sizeof(len));
    std::vector<uint8_t> bytes(len);
    in.read(reinterpret_cast<char*>(bytes.data()), len);
    in.close();
    return bytes;
}
