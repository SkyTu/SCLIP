#ifndef SCLIP_FIX_H
#define SCLIP_FIX_H

#pragma once

#include <cstdint>
#include <cassert>
#include <stdexcept>
#include <type_traits>
#include <iostream>
#include <iomanip> // For std::setprecision
#include <type_traits> // For std::make_signed
#include <Eigen/Dense> // For NumTraits specialization
#include "random.h"

// Forward declaration
template <typename T, int bw, int f, int k>
struct Fix;

// Type trait to check if a type is a specialization of Fix
template<typename>
struct is_fix : std::false_type {};
template<typename T, int bw, int f, int k>
struct is_fix<Fix<T, bw, f, k>> : std::true_type {};


template <typename T, int bw, int f, int k>
std::ostream& operator<<(std::ostream& os, const Fix<T, bw, f, k>& fix);

template <typename T, int bw, int f, int k>
class Fix {
public:
    // Expose underlying data type
    using val_type = T;

    // Expose template parameters as static members
    static constexpr int bitwidth = bw;
    static constexpr int frac_bits = f;
    static constexpr int int_bits = k;

    T val;

    // Default constructor
    Fix() : val(0) {
        static_assert(bw >= (1 + f + k), "Bitwidth too small: require at least 1 + k + f bits");
    }

    // Constructor from underlying integer type
    explicit Fix(T initial_val) : val(initial_val) {}

    // Constructor to cast from another Fix type with a different bitwidth
    template <int other_bw>
    Fix(const Fix<T, other_bw, f, k>& other) : val(other.val) {}

    // Constructor from float
    template <typename TFloat, typename = std::enable_if_t<std::is_floating_point<TFloat>::value>>
    Fix(TFloat float_val) {
        static_assert(bw >= (1 + f + k), "Bitwidth too small: require at least 1 + k + f bits");
        val = static_cast<T>(float_val * (1ULL << f));
    }

    // Constructor from float with custom scale
    template <typename TFloat, typename = std::enable_if_t<std::is_floating_point<TFloat>::value>>
    Fix(TFloat float_val, int custom_f) {
        static_assert(bw >= (1 + f + k), "Bitwidth too small: require at least 1 + k + f bits");
        val = static_cast<T>(float_val * (1ULL << custom_f));
    }
    
    template <typename TFloat>
    TFloat to_float() const {
        T sign_bit_mask = T(1) << (bw - 1);
        double scaled_val;

        if (val & sign_bit_mask) { // Negative
            T inverted = ~val + 1;
            T value_mask = (T(1) << (bw - 1)) - 1;
            if (bw == sizeof(T) * 8) {
                 value_mask = ~(T(1) << (bw - 1));
            }
            if (val == T(-1)) {
                 scaled_val = -1.0;
            } else {
                 scaled_val = -static_cast<double>(inverted & value_mask);
            }
        } else { // Positive
            scaled_val = static_cast<double>(val);
        }
        
        return static_cast<TFloat>(scaled_val / (1.0 * (1ULL << f)));
    }

    Fix<T, bw, f, k> trunc(int shift_bits) const {
        T sign_bit_mask = T(1) << (bw - 1);
        T new_val;
        if (val & sign_bit_mask) { // Negative
            T mask = (~T(0)) << (bw - shift_bits);
            new_val = (val >> shift_bits) | mask;
        } else { // Positive
            new_val = val >> shift_bits;
        }
        return Fix<T, bw, f, k>(new_val);
    }

    Fix<T, bw, f, k> operator+(const Fix<T, bw, f, k>& other) const {
        T result_val = this->val + other.val;
        if (bw < sizeof(T) * 8) {
            T mask = (T(1) << bw) - 1;
            result_val &= mask;
        }
        return Fix<T, bw, f, k>(result_val);
    }
    
    Fix<T, bw, f, k>& operator+=(const Fix<T, bw, f, k>& other) {
        this->val += other.val;
        if (bw < sizeof(T) * 8) {
            T mask = (T(1) << bw) - 1;
            this->val &= mask;
        }
        return *this;
    }

    Fix<T, bw, f, k> operator*(const Fix<T, bw, f, k>& other) const {
        // Widening multiplication to prevent overflow of the raw product.
        // The result remains in 2*f fractional bit format and must be truncated later
        // by a function like trunc() or within a protocol like tensor_mul.
        T temp = val * other.val;
        return Fix<T, bw, f, k>(temp);
    }
    
    Fix<T, bw, f, k> operator-(const Fix<T, bw, f, k>& other) const {
        T result_val = this->val - other.val;
        if (bw < sizeof(T) * 8) {
            T mask = (T(1) << bw) - 1;
            result_val &= mask;
        }
        return Fix<T, bw, f, k>(result_val);
    }

    Fix<T, bw, f, k>& operator-=(const Fix<T, bw, f, k>& other) {
        this->val -= other.val;
        if (bw < sizeof(T) * 8) {
            T mask = (T(1) << bw) - 1;
            this->val &= mask;
        }
        return *this;
    }

    Fix<T, bw, f, k> mul(const Fix<T, bw, f, k>& other) const {
        return Fix<T, bw, f, k>(this->val * other.val);
    }

    Fix<T, bw, f, k> operator-() const { return Fix<T, bw, f, k>(~val + 1); }

    template <int new_bw, int new_f, int new_k>
    Fix<T, new_bw, new_f, new_k> change_format() const {
        // This function changes the fixed-point format of the number.
        // It converts the fixed-point number to a float and then back to the new fixed-point format.
        double float_val = this->template to_float<double>();
        return Fix<T, new_bw, new_f, new_k>(float_val);
    }

    T get_val() const { return val; }

    template<int new_bw, int new_f, int new_k>
    Fix<T, new_bw, new_f, new_k> get_msb() const {
        T msb = (val >> (bw - 1)) & 1;
        return Fix<T, new_bw, new_f, new_k>(msb);
    }
};

template <typename T, int bw, int f, int k>
std::ostream& operator<<(std::ostream& os, const Fix<T, bw, f, k>& fix) {
    os << fix.template to_float<double>();
    return os;
}

namespace Eigen {
    template<typename T, int bw, int f, int k>
    struct NumTraits<Fix<T, bw, f, k>> : GenericNumTraits<Fix<T, bw, f, k>>
    {
        typedef Fix<T, bw, f, k> Real;
        typedef Fix<T, bw, f, k> NonInteger;
        typedef Fix<T, bw, f, k> Nested;

        enum {
            IsComplex = 0,
            IsInteger = 0,
            IsSigned = std::is_signed<T>::value,
            RequireInitialization = 1,
            ReadCost = 1,
            AddCost = 1,
            MulCost = 1
        };

        static inline Real epsilon() { 
            return Fix<T, bw, f, k>(static_cast<T>(1)); 
        }
        static inline Real dummy_precision() { return Real(0); }
        static inline int digits10() { return 0; }
    };
}

#endif // SCLIP_FIX_H

