#ifndef SCLIP_FIX_TENSOR_H
#define SCLIP_FIX_TENSOR_H

#include <unsupported/Eigen/CXX11/Tensor>
#include "mpc/fix.h"
#include "utils/random.h"
#include "utils/io.h"
#include <vector>
#include <string>
#include <cassert>
#include <cstring>
#include <iostream>
#include <iomanip> // For std::setprecision

// Forward declaration
template <typename T, int bw, int f, int k, int Rank, int Options>
class FixTensor;

// ================= Traits =================
namespace Eigen {
namespace internal {
template <typename T, int bw, int f, int k, int Rank, int Options>
struct traits<FixTensor<T, bw, f, k, Rank, Options>> : public traits<Eigen::Tensor<Fix<T, bw, f, k>, Rank, Options>> {
};
} // namespace internal
} // namespace Eigen


// General template definition
template <typename T, int bw, int f, int k, int Rank, int Options = Eigen::RowMajor>
class FixTensor : public Eigen::Tensor<Fix<T, bw, f, k>, Rank, Options> {
public:
    using Base = Eigen::Tensor<Fix<T, bw, f, k>, Rank, Options>;
    using typename Base::Scalar;
    using FixType = Fix<T, bw, f, k>;

    // Inherit constructors from Eigen::Tensor
    using Base::Base;
    
    // This constructor allows construction from other Eigen expressions
    template<typename OtherDerived>
    FixTensor(const Eigen::TensorBase<OtherDerived, 0>& other) : Base(other) {}

    // This assignment operator allows assignment from other Eigen expressions
    template<typename OtherDerived>
    FixTensor& operator=(const Eigen::TensorBase<OtherDerived, 0>& other) {
        this->Base::operator=(other);
        return *this;
    }

    // New in-place truncation method
    void trunc_in_place(int shift_bits) {
        for (long long i = 0; i < this->size(); ++i) {
            this->data()[i] = this->data()[i].trunc(shift_bits);
        }
    }

    void print(const std::string& title = "") const {
        if (!title.empty()) std::cout << title << std::endl;
        std::cout << *this << std::endl;
        std::cout << "(Note: Pretty printing is only supported for ranks 1, 2, and 3)" << std::endl;
    }

    void ones() { this->setConstant(FixType(1.0)); }
    void zeros() { this->setZero(); }

    void initialize(int k_int = 3, int gap_in = -1) {
        assert(k >= k_int && "Input k must be smaller or equal to the k of FixType");
        
        // The gap is the number of sign-extended bits between the sign and the integer part.
        // It's the difference between the original redundant bits (f) and the new redundant bits.
        // New redundant bits = original_redundant_bits + (original_k - new_k) = f + k - k_int
        int gap = (gap_in == -1) ? (f + k - k_int) : gap_in;
        
        Random random_gen;
        T* rand_data = random_gen.randomGEwithGap<T>(this->size(), bw, gap);
        memcpy(this->data(), rand_data, this->size() * sizeof(FixType));
        delete[] rand_data;
    }
};

// Explicit operator overloads for FixTensor to bypass Eigen expression template issues
template <typename T, int bw, int f, int k, int Rank, int Options>
FixTensor<T, bw, f, k, Rank, Options> operator+(const FixTensor<T, bw, f, k, Rank, Options>& a, const FixTensor<T, bw, f, k, Rank, Options>& b) {
    FixTensor<T, bw, f, k, Rank, Options> result(a.dimensions());
    for (long long i = 0; i < a.size(); ++i) {
        result.data()[i] = a.data()[i] + b.data()[i];
    }
    return result;
}

template <typename T, int bw, int f, int k, int Rank, int Options>
FixTensor<T, bw, f, k, Rank, Options> operator-(const FixTensor<T, bw, f, k, Rank, Options>& a, const FixTensor<T, bw, f, k, Rank, Options>& b) {
    FixTensor<T, bw, f, k, Rank, Options> result(a.dimensions());
    for (long long i = 0; i < a.size(); ++i) {
        result.data()[i] = a.data()[i] - b.data()[i];
    }
    return result;
}

template <typename T, int bw, int f, int k, int Rank, int Options>
FixTensor<T, bw, f, k, Rank, Options> operator*(const FixTensor<T, bw, f, k, Rank, Options>& a, const FixTensor<T, bw, f, k, Rank, Options>& b) {
    assert(a.dimensions() == b.dimensions());
    FixTensor<T, bw, f, k, Rank, Options> result(a.dimensions());
    for (long long i = 0; i < a.size(); ++i) {
        result.data()[i] = a.data()[i] * b.data()[i];
    }
    return result;
}

// Specialization for Rank 1
template <typename T, int bw, int f, int k, int Options>
class FixTensor<T, bw, f, k, 1, Options> : public Eigen::Tensor<Fix<T, bw, f, k>, 1, Options> {
public:
    using Base = Eigen::Tensor<Fix<T, bw, f, k>, 1, Options>;
    using FixType = Fix<T, bw, f, k>;
    using Base::Base;
    template<typename OtherDerived>
    FixTensor(const Eigen::TensorBase<OtherDerived, 0>& other) : Base(other) {}

    void trunc_in_place(int shift_bits) {
        for (long long i = 0; i < this->size(); ++i) {
            this->data()[i] = this->data()[i].trunc(shift_bits);
        }
    }

    void print(const std::string& title = "") const {
        if (!title.empty()) std::cout << title << std::endl;
        std::cout << std::fixed << std::setprecision(4);
        std::cout << "[";
        for (long long i = 0; i < this->dimension(0); ++i) {
            std::cout << (*this)(i);
            if (i < this->dimension(0) - 1) std::cout << ", ";
        }
        std::cout << "]";
        std::cout.unsetf(std::ios_base::floatfield);
        std::cout << std::endl;
    }
    void ones() { this->setConstant(FixType(1.0)); }
    void zeros() { this->setZero(); }
    void initialize(int k_int = 3, int gap_in = -1) {
        assert(k >= k_int && "Input k must be smaller or equal to the k of FixType");
        int gap = (gap_in == -1) ? (f + k - k_int) : gap_in;
        Random random_gen;
        T* rand_data = random_gen.randomGEwithGap<T>(this->size(), bw, gap);
        memcpy(this->data(), rand_data, this->size() * sizeof(FixType));
        delete[] rand_data;
    }
};

// Specialization for Rank 2
template <typename T, int bw, int f, int k, int Options>
class FixTensor<T, bw, f, k, 2, Options> : public Eigen::Tensor<Fix<T, bw, f, k>, 2, Options> {
public:
    using Base = Eigen::Tensor<Fix<T, bw, f, k>, 2, Options>;
    using FixType = Fix<T, bw, f, k>;
    using Base::Base;
    template<typename OtherDerived>
    FixTensor(const Eigen::TensorBase<OtherDerived, 0>& other) : Base(other) {}

    void trunc_in_place(int shift_bits) {
        for (long long i = 0; i < this->size(); ++i) {
            this->data()[i] = this->data()[i].trunc(shift_bits);
        }
    }

    void print(const std::string& title = "") const {
        if (!title.empty()) std::cout << title << std::endl;
        std::cout << std::fixed << std::setprecision(4);
        std::cout << "[";
        for (long long i = 0; i < this->dimension(0); ++i) {
            if (i > 0) std::cout << std::endl << " ";
            std::cout << "[";
            for (long long j = 0; j < this->dimension(1); ++j) {
                std::cout << (*this)(i, j);
                if (j < this->dimension(1) - 1) std::cout << ", ";
            }
            std::cout << "]";
        }
        std::cout << "]";
        std::cout.unsetf(std::ios_base::floatfield);
        std::cout << std::endl;
    }
    void ones() { this->setConstant(FixType(1.0)); }
    void zeros() { this->setZero(); }
    void initialize(int k_int = 3, int gap_in = -1) {
        assert(k >= k_int && "Input k must be smaller or equal to the k of FixType");
        int gap = (gap_in == -1) ? (f + k - k_int) : gap_in;
        Random random_gen;
        T* rand_data = random_gen.randomGEwithGap<T>(this->size(), bw, gap);
        memcpy(this->data(), rand_data, this->size() * sizeof(FixType));
        delete[] rand_data;
    }
};

// Specialization for Rank 3
template <typename T, int bw, int f, int k, int Options>
class FixTensor<T, bw, f, k, 3, Options> : public Eigen::Tensor<Fix<T, bw, f, k>, 3, Options> {
public:
    using Base = Eigen::Tensor<Fix<T, bw, f, k>, 3, Options>;
    using FixType = Fix<T, bw, f, k>;
    using Base::Base;
    template<typename OtherDerived>
    FixTensor(const Eigen::TensorBase<OtherDerived, 0>& other) : Base(other) {}
    
    void trunc_in_place(int shift_bits) {
        for (long long i = 0; i < this->size(); ++i) {
            this->data()[i] = this->data()[i].trunc(shift_bits);
        }
    }

    void print(const std::string& title = "") const {
        if (!title.empty()) std::cout << title << std::endl;
        std::cout << std::fixed << std::setprecision(4);
        std::cout << "[";
        for (long long i = 0; i < this->dimension(0); ++i) {
            if (i > 0) std::cout << std::endl << std::endl << " ";
            std::cout << "[";
            for (long long j = 0; j < this->dimension(1); ++j) {
                if (j > 0) std::cout << std::endl << "  ";
                std::cout << "[";
                for (long long l = 0; l < this->dimension(2); ++l) {
                    std::cout << (*this)(i, j, l);
                    if (l < this->dimension(2) - 1) std::cout << ", ";
                }
                std::cout << "]";
            }
            std::cout << "]";
        }
        std::cout << "]";
        std::cout.unsetf(std::ios_base::floatfield);
        std::cout << std::endl;
    }
    void ones() { this->setConstant(FixType(1.0)); }
    void zeros() { this->setZero(); }
    void initialize(int k_int = 3, int gap_in = -1) {
        assert(k >= k_int && "Input k must be smaller or equal to the k of FixType");
        int gap = (gap_in == -1) ? (f + k - k_int) : gap_in;
        Random random_gen;
        T* rand_data = random_gen.randomGEwithGap<T>(this->size(), bw, gap);
        memcpy(this->data(), rand_data, this->size() * sizeof(FixType));
        delete[] rand_data;
    }
};

#endif // SCLIP_FIX_TENSOR_H
