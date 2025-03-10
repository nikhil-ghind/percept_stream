#include <gtest/gtest.h>
#include <vector>
#include <cmath>
#include "preprocess.hpp"
#include "postprocess.hpp"

TEST(PreprocessTest, OutputShape) {
    cv::Mat frame(480, 640, CV_8UC3, cv::Scalar(100, 150, 200));
    auto input = preprocess_frame(frame, 800);
    EXPECT_EQ(input.shape[0], 1);
    EXPECT_EQ(input.shape[1], 3);
    EXPECT_EQ(input.shape[2], 800);
    EXPECT_EQ(input.shape[3], 800);
    EXPECT_EQ((int)input.pixel_values.size(), 1 * 3 * 800 * 800);
}

TEST(PreprocessTest, ValuesInRange) {
    cv::Mat frame(64, 64, CV_8UC3, cv::Scalar(128, 128, 128));
    auto input = preprocess_frame(frame, 64);
    for (float v : input.pixel_values) {
        EXPECT_GE(v, -5.0f);
        EXPECT_LE(v, 5.0f);
    }
}

TEST(PostprocessTest, SingleHighConfidenceDetection) {
    const int num_queries = 10;
    const int num_classes = 5;
    // Construct logits where query 0 has high confidence for class 2
    std::vector<float> logits(num_queries * (num_classes + 1), -10.0f);
    logits[2] = 20.0f;  // query 0, class 2

    // Box: center at (0.5, 0.5) with size (0.2, 0.2)
    std::vector<float> boxes(num_queries * 4, 0.0f);
    boxes[0] = 0.5f; boxes[1] = 0.5f; boxes[2] = 0.2f; boxes[3] = 0.2f;

    auto dets = postprocess(logits.data(), boxes.data(), num_queries, num_classes, 0.3f, 800, 800);
    EXPECT_GE((int)dets.size(), 1);
    EXPECT_EQ(dets[0].class_id, 2);
}
