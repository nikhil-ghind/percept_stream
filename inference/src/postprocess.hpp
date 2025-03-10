#pragma once
#include <vector>
#include <string>
#include <opencv2/opencv.hpp>

struct Detection {
    int class_id;
    float confidence;
    cv::Rect2f box;
};

std::vector<Detection> postprocess(
    const float* logits, const float* pred_boxes,
    int num_queries, int num_classes,
    float score_threshold = 0.5f, int img_w = 800, int img_h = 800);

void draw_detections(cv::Mat& frame, const std::vector<Detection>& dets,
                     const std::vector<std::string>& class_names);
