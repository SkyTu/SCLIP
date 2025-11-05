#include <iostream>

#include "mpc/tensor_ops.h"

// Using a standard C++ integer type, which is much cleaner.
using T = uint64_t;

// 定义测试用的常量
const int BATCH_SIZE = 2;
const int DIM = 4;
const int DUMMY_DIM_0 = 2;

// 从你的代码里复制过来的模板参数
const int IN_BW = 32;
const int F = 16;
const int K_INT = 12;

int main() {
    std::cout << "--- Simple Test for sum_reduce_tensor ---" << std::endl;

    // 1. 直接创建一个3维的FixTensor
    // 形状为 (2, 2, 4)
    FixTensor<T, IN_BW, F, K_INT, 3> input_tensor;
    input_tensor.resize(DUMMY_DIM_0, BATCH_SIZE, DIM);
    input_tensor.setValues({
        {{1.0, 2.0, 3.0, 4.0}, {10.0, 11.0, 12.0, 13.0}},
        {{100.0, 101.0, 102.0, 103.0}, {1000.0, 1001.0, 1002.0, 1003.0}}
    });
    

    std::cout << "\nInput Tensor (data[0]):\n" << input_tensor << std::endl;

    // 2. 调用待测试的函数，沿着axis=2求和
    FixTensor<T, IN_BW, F, K_INT, 2> output_tensor = sum_reduce_tensor<2, T, IN_BW, F, K_INT, Eigen::RowMajor>(input_tensor);

    // 3. 直接打印输出结果的内部数据
    std::cout << "\nOutput Tensor (data[0]):\n" << output_tensor << std::endl;
    
    // 4. 手动计算并展示期望的结果
    std::cout << "\nExpected Output (shape " << DUMMY_DIM_0 << "x" << BATCH_SIZE << "):\n";
    std::cout << (1+2+3+4) << " " << (10+11+12+13) << "\n";
    std::cout << (100+101+102+103) << " " << (1000+1001+1002+1003) << "\n" << std::endl;

    std::cout << "--- Test Complete ---" << std::endl;
    std::cout << "请对比 'Output Tensor' 和 'Expected Output Tensor'." << std::endl;

    return 0;
}
