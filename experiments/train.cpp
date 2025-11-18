#include <iostream>
#include "mpc/mpc.h"
#include "mpc/fix_tensor.h"
#include "mpc/tensor_ops.h"
#include "utils/io.h" // For loading feature data
#include "mpc/truncate.h"
#include "mpc/matmul.h"
#include "utils/random.h"
#include "nn/SGD.h"
#include "nn/FC.h"
#include "nn/Softmax.h"
#include "nn/CosineSimilarity.h"

int main(int argc, char** argv) {
    if(argc != 3){
        std::cerr << "Usage: " << argv[0] << " <party> <dataset_name>" << std::endl;
        return 1;
    }
    int party = std::stoi(argv[1]);
    MPC mpc(2, party);
    std::vector<std::string> addrs = {"127.0.0.1", "127.0.0.1"};
    
    mpc.connect(addrs, 9001);
    
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

    // Add constants for data dimensions, based on test_data_loading.cpp
    const int NUM_TRAIN_SAMPLES = 50000;
    const int NUM_TEST_SAMPLES = 10000;
    const int NUM_CLASSES = 10;

    // Define types for features and labels based on bitwidth
    using FeatureTensor = FixTensor<uint64_t, BW, F, K_INT, 2>;
    // Labels are integers, so they should have 0 fractional bits, as in test_data_loading.cpp
    using LabelTensor = FixTensor<uint64_t, BW, 0, K_INT, 1>;

    // Paths for training data
    std::string dataset_name = argv[2];
    std::string train_feature_dir = "./datasets/" + dataset_name + "/";
    std::string img_feats_path = train_feature_dir + "secure_train_image_feats.bin";
    std::string txt_feats_path = train_feature_dir + "secure_text_feats.bin";
    std::string labels_path = train_feature_dir + "secure_train_labels.bin";
    
    // Paths for validation data
    std::string val_img_feats_path = train_feature_dir + "secure_test_image_feats.bin";
    std::string val_labels_path = train_feature_dir + "secure_test_labels.bin";

    FeatureTensor img_feats, txt_feats, val_img_feats;
    LabelTensor labels, val_labels;

    std::cout << "Party 0: Loading training and validation data..." << std::endl;
    
    // Load training features and labels - strictly follow test_data_loading.cpp logic
    img_feats = load_binary_to_fixtensor<uint64_t, BW, F, K_INT, 2>(img_feats_path, {(long)NUM_TRAIN_SAMPLES, (long)IMAGE_DIM});
    std::cout << "Image features loaded" << std::endl;
    txt_feats = load_binary_to_fixtensor<uint64_t, BW, F, K_INT, 2>(txt_feats_path, {(long)NUM_CLASSES, (long)TEXT_DIM});
    std::cout << "Text features loaded" << std::endl;
    // Load labels with correct type (F=0)
    labels = load_binary_to_fixtensor<uint64_t, BW, 0, K_INT, 1>(labels_path, {(long)NUM_TRAIN_SAMPLES});
    std::cout << "Labels loaded" << std::endl;

    std::cout << "loading validation data..." << std::endl;

    // Load validation features and labels
    val_img_feats = load_binary_to_fixtensor<uint64_t, BW, F, K_INT, 2>(val_img_feats_path, {(long)NUM_TEST_SAMPLES, (long)IMAGE_DIM});
    val_labels = load_binary_to_fixtensor<uint64_t, BW, 0, K_INT, 1>(val_labels_path, {(long)NUM_TEST_SAMPLES});
    

    // Prepare tensors for the batch
    FixTensor<uint64_t, BW, F, K_INT, 2> batch_img_feats_share[(long)(NUM_TRAIN_SAMPLES / BATCH)];
    FixTensor<uint64_t, BW, F, K_INT, 2> batch_txt_feats_share[(long)(NUM_TRAIN_SAMPLES / BATCH)];

    std::cout << (long)(NUM_TRAIN_SAMPLES / BATCH) << " " << BATCH << " " << IMAGE_DIM << std::endl;
    if(mpc.party == 0){
        
        for (int i = 0; i < long(NUM_TRAIN_SAMPLES / BATCH); i++) {
            batch_img_feats_share[i] = FixTensor<uint64_t, BW, F, K_INT, 2>((long)BATCH, (long)IMAGE_DIM);
            batch_txt_feats_share[i] = FixTensor<uint64_t, BW, F, K_INT, 2>((long)BATCH, (long)TEXT_DIM);
        
            // std::cout << "Batch " << i << " started" << std::endl;
            // Manually copy the row for the image feature
            for (int j = 0; j < BATCH; j++) {
                for (int k = 0; k < IMAGE_DIM; k++) {
                    batch_img_feats_share[i](j, k) = img_feats(i * BATCH + j, k);
                }
            }

            // Manually copy the corresponding text feature row using the label as index
            for (int j = 0; j < BATCH; j++) {
                for (int k = 0; k < TEXT_DIM; k++) {
                    batch_txt_feats_share[i](j, k) = txt_feats(labels(i * BATCH + j).val, k);
                }
            }
        }
    }
    else{
        for (int i = 0; i < long(NUM_TRAIN_SAMPLES / BATCH); i++) {
            batch_img_feats_share[i] = FixTensor<uint64_t, BW, F, K_INT, 2>((long)BATCH, (long)IMAGE_DIM);
            batch_txt_feats_share[i] = FixTensor<uint64_t, BW, F, K_INT, 2>((long)BATCH, (long)TEXT_DIM);
            batch_img_feats_share[i].setZero();
            batch_txt_feats_share[i].setZero();
        }
    }
    std::cout << "Data loading complete." << std::endl;
    

    
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
    
    // save randomness
    std::string randomness_path = std::string("./randomness/P") + std::to_string(party) + std::string("/train_") + argv[2] + std::string("_randomness.bin");
    
    // read randomness
    mpc.load_random_data(randomness_path);
    fc_layer_text.read_randomness(mpc);
    fc_layer_image.read_randomness(mpc);
    cos_similarity_layer.read_randomness(mpc);
    softmax_layer.read_randomness(mpc);

    // init weight
    if(mpc.party == 0){
        float w_image_data[IMAGE_DIM * DIM];
        float w_text_data[TEXT_DIM * DIM];
        std::mt19937 gen(42);
        std::uniform_real_distribution<float> dist(-0.01f, 0.01f);
        for(int i = 0; i < IMAGE_DIM * DIM; i++){
            w_image_data[i] = dist(gen);
        }
        
        gen = std::mt19937(195);
        dist = std::uniform_real_distribution<float>(-0.01f, 0.01f);
        for(int i = 0; i < TEXT_DIM * DIM; i++){
            w_text_data[i] = dist(gen);
        }
        uint8_t* weights_image_buffer = new uint8_t[sizeof(w_image_data)];
        uint8_t* weights_text_buffer = new uint8_t[sizeof(w_text_data)];
        memcpy(weights_image_buffer, w_image_data, sizeof(w_image_data));
        memcpy(weights_text_buffer, w_text_data, sizeof(w_text_data));
        std::cout << "randomness generated" << std::endl;
        
        // --- Start of Fix ---
        // Create a temporary pointer to pass to the init function, 
        // preserving the original pointer for deletion.
        uint8_t* temp_image_ptr = weights_image_buffer;
        fc_layer_image.initWeights(&temp_image_ptr, true);
        std::cout << "image weights initialized" << std::endl;

        uint8_t* temp_text_ptr = weights_text_buffer;
        fc_layer_text.initWeights(&temp_text_ptr, true);
        std::cout << "text weights initialized" << std::endl;

        delete[] weights_image_buffer; // Delete the original pointer
        delete[] weights_text_buffer;  // Delete the original pointer
        // --- End of Fix ---
    }
    else{
        // Define zero-filled arrays for the other party
        float w_image_data[IMAGE_DIM * DIM] = {0};
        float w_text_data[TEXT_DIM * DIM] = {0};

        uint8_t* weights_image_buffer = new uint8_t[sizeof(w_image_data)];
        uint8_t* weights_text_buffer = new uint8_t[sizeof(w_text_data)];
        memcpy(weights_image_buffer, w_image_data, sizeof(w_image_data));
        memcpy(weights_text_buffer, w_text_data, sizeof(w_text_data));
 
        // --- Start of Fix ---
        uint8_t* temp_image_ptr = weights_image_buffer;
        fc_layer_image.initWeights(&temp_image_ptr, true);
        std::cout << "image weights initialized" << std::endl;

        uint8_t* temp_text_ptr = weights_text_buffer;
        fc_layer_text.initWeights(&temp_text_ptr, true);
        std::cout << "text weights initialized" << std::endl;

        delete[] weights_image_buffer; // Delete the original pointer
        delete[] weights_text_buffer;  // Delete the original pointer
        // --- End of Fix ---
    }
    std::cout << "weights initialized" << std::endl;
    FixTensor<uint64_t, BW, F, K_INT, 2> masked_img_feats;
    FixTensor<uint64_t, BW, F, K_INT, 2> masked_txt_feats;
    // train
    for(int epoch = 0; epoch < EPOCHS; epoch++){
        std::cout << "Epoch " << epoch << " started" << std::endl;
        for(int i = 0; i < (long)(NUM_TRAIN_SAMPLES / BATCH); i++){
            std::cout << "Training sample " << i << " started" << std::endl;
            // forward pass
            masked_img_feats = batch_img_feats_share[i] - fc_layer_image.randomness.matmul_randomness_fwd.U;
            masked_txt_feats = batch_txt_feats_share[i] - fc_layer_text.randomness.matmul_randomness_fwd.U;
            fc_layer_image.W_rec = fc_layer_image.W_share - fc_layer_image.randomness.matmul_randomness_fwd.V;
            fc_layer_text.W_rec = fc_layer_text.W_share - fc_layer_text.randomness.matmul_randomness_fwd.V;
            
            reconstruct_tensor_parallel(masked_img_feats, masked_txt_feats, fc_layer_image.W_rec, fc_layer_text.W_rec);
            
            auto output_image = fc_layer_image.forward<0>(batch_img_feats_share[i], masked_img_feats);
            auto output_text = fc_layer_text.forward<0>(batch_txt_feats_share[i], masked_txt_feats);

            std::cout << "fc finished" << std::endl;

            auto output_cosine = cos_similarity_layer.forward(output_image, output_text);
            auto cos_tmp = reconstruct_tensor(output_cosine);
            std::cout << "output cosine reconstructed" << std::endl;
            for(int i = 0; i < BATCH; i++){
                std::cout << cos_tmp(i, i) << " ";
                if (i == BATCH - 1){
                    std::cout << std::endl;
                }
            }
            // backward pass
            auto output_softmax = softmax_layer.backward(output_cosine);
            auto output_softmax_rec = reconstruct_tensor(output_softmax);
            std::cout << "output softmax reconstructed" << std::endl;
            for(int i = 0; i < BATCH; i++){
                std::cout << output_softmax_rec(i, i) << " ";
                if (i == BATCH - 1){
                    std::cout << std::endl;
                }
            }
            softmax_layer.update();            
            auto [dI_share, dT_share] = cos_similarity_layer.backward(output_softmax);
            auto dI_rec = reconstruct_tensor(dI_share);
            auto dT_rec = reconstruct_tensor(dT_share);
            std::cout << "dI reconstructed" << std::endl;
            for(int i = 0; i < BATCH; i++){
                std::cout << dI_rec(i, i) << " ";
                if (i == BATCH - 1){
                    std::cout << std::endl;
                }
            }
            auto dI_share_rec = dI_share - fc_layer_image.randomness.matmul_randomness_bwd.U;
            auto dT_share_rec = dT_share - fc_layer_text.randomness.matmul_randomness_bwd.U;
            reconstruct_tensor_parallel(dI_share_rec, dT_share_rec);
            auto grad_output_text = fc_layer_text.backward(dT_share_rec, dT_share);
            
            auto grad_output_image_rec = reconstruct_tensor(grad_output_text);
            std::cout << "grad output image reconstructed" << std::endl;
            for(int i = 0; i < BATCH; i++){
                std::cout << grad_output_image_rec(i, i) << " ";
                if (i == BATCH - 1){
                    std::cout << std::endl;
                }
            }

            fc_layer_text.update(LR);
            auto tmp_weight_text = reconstruct_tensor(fc_layer_text.W_share);
            std::cout << "weight text reconstructed" << std::endl;
            for(int i = 0; i < TEXT_DIM; i++){
                std::cout << tmp_weight_text(i, i) << " ";
                if (i == TEXT_DIM - 1){
                    std::cout << std::endl;
                }
            }
            auto grad_output_image = fc_layer_image.backward(dI_share_rec, dI_share);
            fc_layer_image.update(LR);
        }
        std::cout << "Epoch " << epoch << " completed" << std::endl;
    }

    return 0;
}