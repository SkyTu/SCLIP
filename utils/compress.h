#pragma once

#include <vector>
#include <cstdint>
#include <stdexcept>
#include "mpc/fix.h" // Make sure Fix is defined for the type trait

template<typename T>
std::vector<uint8_t> pack_data(const T* data, size_t num_elements, int bitwidth) {
    size_t total_bits = num_elements * bitwidth;
    size_t total_bytes = (total_bits + 7) / 8;
    std::vector<uint8_t> packed(total_bytes, 0);

    for (size_t i = 0; i < num_elements; ++i) {
        for (int j = 0; j < bitwidth; ++j) {
            size_t bit_pos = i * bitwidth + j;
            size_t byte_pos = bit_pos / 8;
            size_t bit_in_byte = bit_pos % 8;
            
            uint64_t val_to_pack;
            if constexpr (is_fix<T>::value) {
                val_to_pack = data[i].val;
            } else {
                val_to_pack = data[i];
            }

            if ((val_to_pack >> j) & 1) {
                packed[byte_pos] |= (1 << bit_in_byte);
            }
        }
    }
    return packed;
}

template<typename T>
std::vector<T> unpack_data(const std::vector<uint8_t>& packed_data, int num_elements, int bitwidth) {
    if (num_elements == 0) return {};
    std::vector<T> data(num_elements);
    size_t total_bits = num_elements * bitwidth;
    size_t total_bytes = (total_bits + 7) / 8;

    if (packed_data.size() < total_bytes) {
        throw std::runtime_error("Not enough data in packed_data to unpack.");
    }

    for (int i = 0; i < num_elements; ++i) {
        T val = 0;
        for (int j = 0; j < bitwidth; ++j) {
            size_t bit_pos = i * bitwidth + j;
            if ((packed_data[bit_pos / 8] >> (bit_pos % 8)) & 1) {
                val |= (static_cast<T>(1) << j);
            }
        }
        data[i] = val;
    }
    return data;
}
