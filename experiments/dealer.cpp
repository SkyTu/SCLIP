#include <iostream>
#include "mpc/mpc.h"
#include "mpc/fix_tensor.h"
#include "mpc/tensor_ops.h"
#include "mpc/truncate.h"
#include "mpc/matmul.h"
#include "utils/random.h"
#include "nn/SGD.h"
#include "nn/FC.h"
#include "nn/Softmax.h"
#include "nn/CosineSimilarity.h"

int main(int argc, char** argv) {
    
    const int BATCH = 64;
    const int DIM = 512;
    const int TEXT_DIM = 512;
    const int IMAGE_DIM = 512;
    const int EPOCHS = 10;    // 针对CIFAR100增加训练周期
    const int WORLD = 2;
    const int BW = 64;
    const int F = 24;
    const int smallBW = BW - F;
    const int K_INT = BW - 2 - 2 * F;

    
    if(strcmp(argv[1], "CIFAR10") == 0 || strcmp(argv[2], "cifar10") == 0){
        const int BATCH = 64;
        const int TEXT_DIM = 512;
        const int IMAGE_DIM = 512;
        const int DIM = 512;
        const int EPOCHS = 10;
        const int WORLD = 2;
        const int BW = 64;
        const int F = 24;

        const int smallBW = BW - F;
        const int K_INT = BW - 2 - 2 * F;
    }
    
    
    // initialize layers
    FCLayerParams fc_layer_params_text;
    fc_layer_params_text.B = BATCH;
    fc_layer_params_text.in_dim = TEXT_DIM;
    fc_layer_params_text.out_dim = DIM;
    fc_layer_params_text.use_bias = false;
    fc_layer_params_text.reconstructed_input = true;
    FCLayerParams fc_layer_params_image;
    fc_layer_params_image.B = BATCH;
    fc_layer_params_image.in_dim = IMAGE_DIM;
    fc_layer_params_image.out_dim = DIM;
    fc_layer_params_image.use_bias = false;
    fc_layer_params_image.reconstructed_input = true;
    CosSimLayerParams cos_similarity_layer_params;
    cos_similarity_layer_params.B = BATCH;
    cos_similarity_layer_params.in_dim = DIM;
    SoftmaxLayerParams softmax_layer_params;
    softmax_layer_params.B = BATCH;

    FCLayer<uint64_t, BW, smallBW, F, K_INT> fc_layer_text(fc_layer_params_text);
    FCLayer<uint64_t, BW, smallBW, F, K_INT> fc_layer_image(fc_layer_params_image);
    CosSimLayer<uint64_t, smallBW, BW, F, K_INT> cos_similarity_layer(cos_similarity_layer_params);
    SoftmaxLayer<uint64_t, BW, smallBW, F, K_INT> softmax_layer(softmax_layer_params);
    

    // get randomness size
    size_t total_size = 0;
    total_size += fc_layer_text.getRandomnessSize();
    total_size += fc_layer_image.getRandomnessSize();
    total_size += cos_similarity_layer.getRandomnessSize();
    total_size += softmax_layer.getRandomnessSize();
    std::cout << "Total randomness size: " << total_size << std::endl;
    
    uint8_t* p0_data = new uint8_t[total_size];
    uint8_t* p1_data = new uint8_t[total_size];
    Buffer p0_buf(p0_data);
    Buffer p1_buf(p1_data);
    // generate randomness
    fc_layer_text.generate_randomness(p0_buf, p1_buf);
    fc_layer_image.generate_randomness(p0_buf, p1_buf);
    cos_similarity_layer.generate_randomness(p0_buf, p1_buf);
    softmax_layer.generate_randomness(p0_buf, p1_buf);
    
    // save randomness
    std::string p0_randomness_path = std::string("./randomness/P0/train_") + argv[1] + "_randomness.bin";
    std::string p1_randomness_path = std::string("./randomness/P1/train_") + argv[1] + "_randomness.bin";
    std::ofstream p0_randomness_file(p0_randomness_path, std::ios::binary);
    p0_randomness_file.write(reinterpret_cast<const char*>(p0_data), total_size);
    p0_randomness_file.close();
    std::ofstream p1_randomness_file(p1_randomness_path, std::ios::binary);
    p1_randomness_file.write(reinterpret_cast<const char*>(p1_data), total_size);
    p1_randomness_file.close();

    delete[] p0_data;
    delete[] p1_data;

    std::cout << "Dealer finished successfully." << std::endl;
    return 0;
}