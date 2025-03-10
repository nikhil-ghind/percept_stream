#include "preprocess.hpp"
#include <cmath>

static const float MEAN[3] = {0.485f, 0.456f, 0.406f};
static const float STD[3]  = {0.229f, 0.224f, 0.225f};

PreprocessedInput preprocess_frame(const cv::Mat& frame, int target_size) {
    cv::Mat resized;
    cv::resize(frame, resized, cv::Size(target_size, target_size), 0, 0, cv::INTER_LINEAR);

    cv::Mat rgb;
    cv::cvtColor(resized, rgb, cv::COLOR_BGR2RGB);
    rgb.convertTo(rgb, CV_32FC3, 1.0f / 255.0f);

    int H = target_size, W = target_size;
    std::vector<float> pixel_values(1 * 3 * H * W);
    for (int c = 0; c < 3; ++c) {
        for (int y = 0; y < H; ++y) {
            for (int x = 0; x < W; ++x) {
                float v = rgb.at<cv::Vec3f>(y, x)[c];
                pixel_values[c * H * W + y * W + x] = (v - MEAN[c]) / STD[c];
            }
        }
    }

    std::vector<int64_t> pixel_mask(H * W, 1LL);
    return {pixel_values, pixel_mask, {1, 3, H, W}};
}
