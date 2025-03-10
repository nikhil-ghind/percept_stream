#include "detector.hpp"
#include <stdexcept>

Detector::Detector(const std::string& model_path, int num_threads)
    : env_(ORT_LOGGING_LEVEL_WARNING, "detector"),
      session_(nullptr)
{
    opts_.SetIntraOpNumThreads(num_threads);
    opts_.SetGraphOptimizationLevel(ORT_ENABLE_ALL);
    session_ = Ort::Session(env_, model_path.c_str(), opts_);
}

std::pair<std::vector<float>, std::vector<float>> Detector::run(const PreprocessedInput& input) {
    Ort::MemoryInfo mem_info = Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault);

    std::array<int64_t, 4> pv_shape = input.shape;
    std::array<int64_t, 3> pm_shape = {1, input.shape[2], input.shape[3]};

    auto pv_tensor = Ort::Value::CreateTensor<float>(mem_info,
        const_cast<float*>(input.pixel_values.data()), input.pixel_values.size(),
        pv_shape.data(), 4);
    auto pm_tensor = Ort::Value::CreateTensor<int64_t>(mem_info,
        const_cast<int64_t*>(input.pixel_mask.data()), input.pixel_mask.size(),
        pm_shape.data(), 3);

    std::vector<Ort::Value> inputs;
    inputs.push_back(std::move(pv_tensor));
    inputs.push_back(std::move(pm_tensor));

    const char* input_names[] = {"pixel_values", "pixel_mask"};
    const char* output_names[] = {"logits", "pred_boxes"};

    auto outputs = session_.Run(Ort::RunOptions{nullptr}, input_names, inputs.data(), 2, output_names, 2);

    const float* logits_ptr = outputs[0].GetTensorData<float>();
    const float* boxes_ptr = outputs[1].GetTensorData<float>();
    auto logits_shape = outputs[0].GetTensorTypeAndShapeInfo().GetShape();
    auto boxes_shape = outputs[1].GetTensorTypeAndShapeInfo().GetShape();

    size_t logits_size = 1;
    for (auto d : logits_shape) logits_size *= d;
    size_t boxes_size = 1;
    for (auto d : boxes_shape) boxes_size *= d;

    return {
        std::vector<float>(logits_ptr, logits_ptr + logits_size),
        std::vector<float>(boxes_ptr, boxes_ptr + boxes_size)
    };
}
