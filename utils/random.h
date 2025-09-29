#pragma once

#include <cstdint>
#include <stdexcept>
#include <vector>
#include <random>
#include <limits>

class Random {
private:
    std::mt19937_64 rng;

public:
    Random() {
        std::random_device rd;
        rng.seed(rd());
    }

    template<typename T>
    T* randomGE(const uint64_t n, int bw) {
        T* data = new T[n];
        std::uniform_int_distribution<T> distrib(0, std::numeric_limits<T>::max());
        
        T mask = (static_cast<T>(1) << (bw - 1));
        mask |= (mask - 1);
        if (bw == sizeof(T) * 8) {
            mask = std::numeric_limits<T>::max();
        }

        for (uint64_t i = 0; i < n; ++i) {
            data[i] = distrib(rng) & mask;
        }
        return data;
    }

    template <typename T>
    T* random_ge_c(int n, int bitwidth) {
        if (bitwidth > 64) {
            throw std::runtime_error("Bitwidth cannot be greater than 64.");
        }
        T* data = new T[n];
        std::uniform_int_distribution<T> distrib(0, std::numeric_limits<T>::max());
        
        T mask = (static_cast<T>(1) << (bitwidth - 1));
        mask |= (mask - 1);
        if (bitwidth == sizeof(T) * 8) {
            mask = std::numeric_limits<T>::max();
        }

        for (int i = 0; i < n; ++i) {
            data[i] = distrib(rng) & mask;
        }
        return data;
    }

    template <typename T>
    T* randomGEwithGap(const uint64_t n, int bw, int gap) {
        if (bw - 1 - gap <= 0) {
            throw std::runtime_error("Bitwidth after gap is not positive.");
        }

        T* data = new T[n];
        std::uniform_int_distribution<T> distrib(0, std::numeric_limits<T>::max());
        
        int rand_bits_len = bw - 1 - gap;
        T rand_mask = (rand_bits_len >= sizeof(T) * 8) ? std::numeric_limits<T>::max() : (static_cast<T>(1) << rand_bits_len) - 1;
        
        for (uint64_t i = 0; i < n; ++i) {
            T rand_part = distrib(rng) & rand_mask;
            T sign_bit = (distrib(rng) % 2); // 0 for positive, 1 for negative

            if (sign_bit == 0) { // Positive
                data[i] = rand_part;
            } else { // Negative
                // Sign extend all bits above the random part
                T sign_extension_mask = (~T(0)) << rand_bits_len;
                data[i] = sign_extension_mask | rand_part;
            }
        }
        return data;
    }
};

