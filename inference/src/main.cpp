#include <iostream>
#include <fstream>
#include <string>
#include <vector>
#include <chrono>
#include <opencv2/opencv.hpp>
#include "detector.hpp"
#include "preprocess.hpp"
#include "postprocess.hpp"

int main(int argc, char** argv) {
    std::string model_path = "models/model_int8.onnx";
    std::string source = "0";
    std::string labels_path = "models/labels.txt";
    float threshold = 0.5f;
    int num_classes = 80;

    for (int i = 1; i < argc; ++i) {
        std::string arg = argv[i];
        if (arg == "--model" && i + 1 < argc) model_path = argv[++i];
        else if (arg == "--source" && i + 1 < argc) source = argv[++i];
        else if (arg == "--labels" && i + 1 < argc) labels_path = argv[++i];
        else if (arg == "--threshold" && i + 1 < argc) threshold = std::stof(argv[++i]);
    }

    std::vector<std::string> class_names;
    std::ifstream lf(labels_path);
    if (lf) { std::string line; while (std::getline(lf, line)) class_names.push_back(line); }

    cv::VideoCapture cap(source == "0" ? cv::VideoCapture(0) : cv::VideoCapture(source));
    if (!cap.isOpened()) { std::cerr << "Cannot open source: " << source << "\n"; return 1; }

    Detector detector(model_path);
    cv::Mat frame;

    while (true) {
        if (!cap.read(frame) || frame.empty()) break;
        auto t0 = std::chrono::high_resolution_clock::now();

        auto input = preprocess_frame(frame);
        auto [logits, boxes] = detector.run(input);
        auto dets = postprocess(logits.data(), boxes.data(), 100, num_classes, threshold, frame.cols, frame.rows);
        draw_detections(frame, dets, class_names);

        auto t1 = std::chrono::high_resolution_clock::now();
        double ms = std::chrono::duration<double, std::milli>(t1 - t0).count();
        double fps = 1000.0 / ms;
        cv::putText(frame, "FPS: " + std::to_string((int)fps), {10, 30},
                    cv::FONT_HERSHEY_SIMPLEX, 1.0, {0, 255, 255}, 2);

        cv::imshow("Detections", frame);
        if (cv::waitKey(1) == 'q') break;
    }
    return 0;
}
