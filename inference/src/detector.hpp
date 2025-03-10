#pragma once
#include <string>
#include <vector>
#include <utility>
#include <onnxruntime/core/providers/cpu/cpu_execution_provider.h>
#include <onnxruntime_cxx_api.h>
#include "preprocess.hpp"

class Detector {
public:
    explicit Detector(const std::string& model_path, int num_threads = 4);
    std::pair<std::vector<float>, std::vector<float>> run(const PreprocessedInput& input);

private:
    Ort::Env env_;
    Ort::SessionOptions opts_;
    Ort::Session session_;
};
