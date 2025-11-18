#include <iostream>
#include <vector>
#include <string>
#include <numeric>
#include <algorithm>
#include <random>

#include "mpc/fix_tensor.h"
#include "utils/io.h"

// Define constants for the test
const int NUM_TRAIN_SAMPLES = 50000;
const int NUM_CLASSES = 10;
const int IMAGE_DIM = 512;
const int TEXT_DIM = 512; // Should be same as IMAGE_DIM for feature space
const int BATCH_SIZE = 64;

// Use the same template parameters as in train.cpp for consistency
const int BW = 64;
const int smallBW = 48;
const int F = 24;
const int K_INT = BW - 2 - 2 * F;

using ImageFeatureTensor = FixTensor<uint64_t, smallBW, F, K_INT, 2>;
using TextFeatureTensor = FixTensor<uint64_t, smallBW, F, K_INT, 2>;
// Labels are integer values, so we use a FixTensor with 0 fractional bits.
// And use the larger bitwidth as requested.
using LabelTensor = FixTensor<uint64_t, BW, 0, K_INT, 1>;


int main(int argc, char** argv) {
    if (argc != 2) {
        std::cerr << "Usage: " << argv[0] << " <dataset_name>" << std::endl;
        return 1;
    }

    std::string dataset_name = argv[1];
    std::cout << "--- Testing Data Loading and Batch Creation for " << dataset_name << " ---" << std::endl;

    // 1. Load the full datasets
    std::cout << "Loading full datasets..." << std::endl;
    std::string feature_dir = "./datasets/" + dataset_name + "/";
    
    ImageFeatureTensor all_img_feats = load_binary_to_fixtensor<uint64_t, smallBW, F, K_INT, 2>(
        feature_dir + "secure_train_image_feats.bin", 
        {(long)NUM_TRAIN_SAMPLES, (long)IMAGE_DIM}
    );

    TextFeatureTensor all_txt_feats = load_binary_to_fixtensor<uint64_t, smallBW, F, K_INT, 2>(
        feature_dir + "secure_text_feats.bin", 
        {(long)NUM_CLASSES, (long)TEXT_DIM}
    );

    LabelTensor all_labels = load_binary_to_fixtensor<uint64_t, BW, 0, K_INT, 1>(
        feature_dir + "secure_train_labels.bin", 
        {(long)NUM_TRAIN_SAMPLES}
    );
    
    std::cout << "Full image features shape: " << all_img_feats.dimension(0) << "x" << all_img_feats.dimension(1) << std::endl;
    std::cout << "Full text features shape:  " << all_txt_feats.dimension(0) << "x" << all_txt_feats.dimension(1) << std::endl;
    std::cout << "Full labels shape:         " << all_labels.dimension(0) << std::endl;
    std::cout << "Datasets loaded successfully.\n" << std::endl;

    // 2. Create a balanced batch
    std::cout << "Creating one balanced batch of size " << BATCH_SIZE << "..." << std::endl;

    // Create a vector of indices [0, 1, 2, ..., 49999] and shuffle it
    std::vector<int> indices(NUM_TRAIN_SAMPLES);
    std::iota(indices.begin(), indices.end(), 0);
    std::random_device rd;
    std::mt19937 g(rd());
    std::shuffle(indices.begin(), indices.end(), g);

    // Prepare tensors for the batch
    Eigen::array<long, 2> img_dims = {(long)BATCH_SIZE, (long)IMAGE_DIM};
    ImageFeatureTensor batch_img_feats(img_dims);

    Eigen::array<long, 2> txt_dims = {(long)BATCH_SIZE, (long)TEXT_DIM};
    TextFeatureTensor batch_txt_feats(txt_dims);
    
    Eigen::array<long, 1> label_dims = {(long)BATCH_SIZE};
    LabelTensor batch_labels(label_dims);

    for (int i = 0; i < BATCH_SIZE; ++i) {
        int sample_idx = indices[i];
        
        // Get the label for the current sample.
        int label_val = static_cast<int>(all_labels.data()[sample_idx].val);
        batch_labels.data()[i] = all_labels.data()[sample_idx];

        // Manually copy the row for the image feature
        for (int j = 0; j < IMAGE_DIM; ++j) {
            batch_img_feats.data()[i * IMAGE_DIM + j] = all_img_feats.data()[sample_idx * IMAGE_DIM + j];
        }

        // Manually copy the corresponding text feature row using the label as index
        for (int j = 0; j < TEXT_DIM; ++j) {
            batch_txt_feats.data()[i * TEXT_DIM + j] = all_txt_feats.data()[label_val * TEXT_DIM + j];
        }
    }

    std::cout << "Batch creation complete." << std::endl;
    std::cout << "Batch image features shape: " << batch_img_feats.dimension(0) << "x" << batch_img_feats.dimension(1) << std::endl;
    std::cout << "Batch text features shape:  " << batch_txt_feats.dimension(0) << "x" << batch_txt_feats.dimension(1) << std::endl;
    std::cout << "Batch labels shape:         " << batch_labels.dimension(0) << std::endl;

    // 3. Verification
    std::cout << "\nVerifying a few samples..." << std::endl;
    for (int i = 0; i < 3; ++i) { // Check the first 3 samples of the batch
        int original_sample_idx = indices[i];
        int label = static_cast<int>(batch_labels.data()[i].val);
        std::cout << "  Sample " << i << " in batch:" << std::endl;
        std::cout << "    - Corresponds to original sample index: " << original_sample_idx << std::endl;
        std::cout << "    - Has label: " << label << std::endl;
        
        // A simple check: does the first element of the batch text feature match the first element of the text feature at the label's index?
        bool is_match = (batch_txt_feats.data()[i * TEXT_DIM].val == all_txt_feats.data()[label * TEXT_DIM].val);
        if (is_match) {
            std::cout << "    - Verification PASSED: Batch text feature matches the one from the full text feature set for this label." << std::endl;
        } else {
            std::cout << "    - Verification FAILED!" << std::endl;
        }
    }

    return 0;
}
