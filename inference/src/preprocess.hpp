#pragma once
#include <vector>
#include <array>
#include <cstdint>
#include <opencv2/opencv.hpp>

struct PreprocessedInput {
    std::vector<float> pixel_values;
    std::vector<int64_t> pixel_mask;
    std::array<int64_t, 4> shape;
};

PreprocessedInput preprocess_frame(const cv::Mat& frame, int target_size = 800);
