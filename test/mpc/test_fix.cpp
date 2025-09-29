#include "fix.h"
#include <iostream>
#include <string>
#include <algorithm>
#include <vector>
#include <cmath>

// Needed for __uint128_t printing
std::ostream& operator<<(std::ostream& os, __uint128_t val) {
    if (val == 0) return os << "0";
    std::string s = "";
    while (val > 0) {
        s += (val % 10) + '0';
        val /= 10;
    }
    std::reverse(s.begin(), s.end());
    return os << s;
}

void test_constructors_and_conversion() {
    std::cout << "--- Testing Constructors and float conversion ---" << std::endl;
    
    float val1 = 3.14159f;
    Fix<uint64_t> f1(val1, 64, 20, 23);
    float val1_back = f1.to_float<float>();
    std::cout << val1 << " -> " << f1.val << " -> " << val1_back << std::endl;
    assert(std::fabs(val1 - val1_back) < 1e-5);

    float val2 = -3.14159f;
    Fix<uint64_t> f2(val2, 64, 20, 23);
    float val2_back = f2.to_float<float>();
    std::cout << val2 << " -> " << f2.val << " -> " << val2_back << std::endl;
    assert(std::fabs(val2 - val2_back) < 1e-5);

    // Test with 128 bit
    double large_double = 1.23456789e18;
    Fix<__uint128_t> f3(large_double, 128, 60, 7); // 1 + 60*2 + 7 = 128
    double large_double_back = f3.to_float<double>();
    std::cout << large_double << " -> " << f3.val << " -> " << large_double_back << std::endl;
    assert(std::fabs(large_double - large_double_back) / large_double < 1e-10); // Use relative error for large numbers
}

void test_trunc_and_change_bw() {
    std::cout << "\n--- Testing trunc and change_bw ---" << std::endl;
    
    // Test trunc
    Fix<uint16_t> f_trunc_pos(1000, 16, 4, 7);
    Fix<uint16_t> f_trunc_pos_res = f_trunc_pos.trunc(4);
    std::cout << f_trunc_pos.val << " >> 4 = " << f_trunc_pos_res.val << std::endl;
    assert(f_trunc_pos_res.val == 62);
    
    Fix<uint16_t> f_trunc_neg(static_cast<uint16_t>(-1000), 16, 4, 7);
    Fix<uint16_t> f_trunc_neg_res = f_trunc_neg.trunc(4);
    std::cout << static_cast<int16_t>(f_trunc_neg.val) << " >> 4 = " << static_cast<int16_t>(f_trunc_neg_res.val) << std::endl;
    assert(static_cast<int16_t>(f_trunc_neg_res.val) == -63);
    
    // Test change_bw
    Fix<uint16_t> f_change_bw(-10.5f, 16, 4, 7);
    Fix<uint32_t> f_expanded = f_change_bw.change_bw<uint32_t>(32, 8, 15);
    float f_expanded_float = f_expanded.to_float<float>();
    std::cout << "Expand Negative: " << -10.5f << " -> " << f_expanded_float << std::endl;
    assert(std::fabs(-10.5f - f_expanded_float) < 1e-5);
    Fix<uint32_t> f_decreased = f_expanded.change_bw<uint32_t>(16, 4, 3);
    float f_decreased_float = f_decreased.to_float<float>();
    std::cout << "Decrease Negative: " << f_expanded_float << " -> " << f_decreased_float << std::endl;
    assert(std::fabs(-10.5f - f_decreased_float) < 1e-5);
}

void test_operators() {
    std::cout << "\n--- Testing Operators ---" << std::endl;
    Fix<uint8_t> a(10, 8, 2, 3);
    Fix<uint8_t> b(20, 8, 2, 3);
    Fix<uint8_t> c = a + b;
    std::cout << int(a.val) << " + " << int(b.val) << " = " << int(c.val) << std::endl;
    assert(c.val == 30);

    Fix<uint32_t> x(200, 32, 2, 3);
    Fix<uint32_t> y(100, 32, 2, 3);
    Fix<uint32_t> z = x + y;
    std::cout << int(x.val) << " + " << int(y.val) << " = " << int(z.val) << " (mod 256)" << std::endl;
    assert(z.val == 300);

    Fix<uint32_t> m2 = x * y;
    assert(m2.val == 20000);
    std::cout << int(x.val) << " * " << int(y.val) << " = " << int(m2.val) << " (mod 256)" << std::endl;
    m2 = m2.trunc(2);
    std::cout << x.to_float<float>() << " " << y.to_float<float>() << " " << m2.val << " " << m2.to_float<float>() << std::endl;
    assert(std::fabs(x.to_float<float>() * y.to_float<float>() - m2.to_float<float>()) < 1e-5);

}

int main() {
    try {
        test_constructors_and_conversion();
        test_trunc_and_change_bw();
        test_operators();
    } catch (const std::exception& e) {
        std::cerr << "An exception occurred: " << e.what() << std::endl;
        return 1;
    }
    return 0;
}
