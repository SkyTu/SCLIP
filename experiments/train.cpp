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
#include "nn/CosSimilarity.h"

int main(int argc, char** argv) {
    if(argc != 2){
        std::cerr << "Usage: " << argv[0] << " <party>" << std::endl;
        return 1;
    }
    int party = std::stoi(argv[1]);
    MPC mpc(party);
    mpc.connect(addrs, port_offset);
    
    int BATCH = 64;
    int DIM = 512;
    int TEXT_DIM = 768;
    int IMAGE_DIM = 512;
    double LR = 1e-3;  // 针对CIFAR100降低学习率
    int EPOCHS = 10;    // 针对CIFAR100增加训练周期
    int WORLD = 2;
    int BW = 64;
    int F = 24;
    int K_INT = BW - 2 - 2 * F;
    if(argv[2] == "CIFAR10" || "cifar10"){
        BATCH = 64;
        TEXT_DIM = 768;
        IMAGE_DIM = 512;
        DIM = 512;
        LR = 1e-3;  // 针对CIFAR100降低学习率
        EPOCHS = 10;    // 针对CIFAR100增加训练周期
        WORLD = 2;
        BW = 64;
        F = 24;
        K_INT = BW - 2 - 2 * F;
    }
    
    // initialize layers
    FCLayerParams fc_layer_params_text(BATCH, TEXT_DIM, DIM, false, true);
    FCLayerParams fc_layer_params_image(BATCH, IMAGE_DIM, DIM, false, true);
    CosSimilarityLayerParams cos_similarity_layer_params(BATCH, DIM);
    SoftmaxLayerParams softmax_layer_params(BATCH);
    FCLayer<float, BW, F, K_INT> fc_layer_text(fc_layer_params_text);
    FCLayer<float, BW, F, K_INT> fc_layer_image(fc_layer_params_image);
    CosSimilarityLayer<float, BW, F, K_INT> cos_similarity_layer(cos_similarity_layer_params);
    SoftmaxLayer<float, BW, F, K_INT> softmax_layer(softmax_layer_params);
    

    // get randomness size
    size_t total_size = 0;
    total_size += fc_layer_text.getRandomnessSize();
    total_size += fc_layer_image.getRandomnessSize();
    total_size += cos_similarity_layer.getRandomnessSize();
    total_size += softmax_layer.getRandomnessSize();
    std::cout << "Total randomness size: " << total_size << std::endl;
    
    Buffer p0_buf(total_size);
    Buffer p1_buf(total_size);
    // generate randomness
    fc_layer.generate_randomness(p0_buf, p1_buf);
    cos_similarity_layer.generate_randomness(p0_buf, p1_buf);
    softmax_layer.generate_randomness(p0_buf, p1_buf);
    
    // read randomness
    fc_layer.readForwardRandomness(mpc);
    fc_layer.readBackwardRandomness(mpc);
    cos_similarity_layer.readForwardRandomness(mpc);
    cos_similarity_layer.readBackwardRandomness(mpc);
    return 0;
}