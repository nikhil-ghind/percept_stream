#include "postprocess.hpp"
#include <cmath>
#include <algorithm>
#include <numeric>

static float softmax_max(const float* logits, int n, int& argmax_idx) {
    float max_val = *std::max_element(logits, logits + n);
    float sum = 0.0f;
    for (int i = 0; i < n; ++i) sum += std::exp(logits[i] - max_val);
    float max_prob = std::exp(logits[0] - max_val) / sum;
    argmax_idx = 0;
    for (int i = 1; i < n; ++i) {
        float p = std::exp(logits[i] - max_val) / sum;
        if (p > max_prob) { max_prob = p; argmax_idx = i; }
    }
    return max_prob;
}

std::vector<Detection> postprocess(
    const float* logits, const float* pred_boxes,
    int num_queries, int num_classes,
    float score_threshold, int img_w, int img_h)
{
    std::vector<Detection> dets;
    for (int q = 0; q < num_queries; ++q) {
        const float* q_logits = logits + q * (num_classes + 1);
        int class_id = 0;
        float conf = softmax_max(q_logits, num_classes, class_id);
        if (class_id == num_classes) continue;  // background
        if (conf < score_threshold) continue;

        const float* box = pred_boxes + q * 4;
        float cx = box[0] * img_w;
        float cy = box[1] * img_h;
        float w  = box[2] * img_w;
        float h  = box[3] * img_h;
        dets.push_back({class_id, conf, cv::Rect2f(cx - w/2, cy - h/2, w, h)});
    }
    return dets;
}

void draw_detections(cv::Mat& frame, const std::vector<Detection>& dets,
                     const std::vector<std::string>& class_names)
{
    for (const auto& d : dets) {
        cv::Rect r(d.box);
        cv::rectangle(frame, r, cv::Scalar(0, 255, 0), 2);
        std::string label = (d.class_id < (int)class_names.size() ? class_names[d.class_id] : std::to_string(d.class_id))
                          + " " + std::to_string((int)(d.confidence * 100)) + "%";
        cv::putText(frame, label, cv::Point(r.x, r.y - 5), cv::FONT_HERSHEY_SIMPLEX, 0.5, cv::Scalar(0,255,0), 1);
    }
}
