# Percept Stream

DETR fine-tuning pipeline with ONNX export, INT8 quantization, and a C++ ONNX Runtime inference application capable of 30+ FPS real-time detection.

## Overview

- **Fine-tune** `facebook/detr-resnet-50` on COCO-format dataset with mixed-precision training
- **Export** to ONNX FP32 via `torch.onnx.export`, validate PyTorch ↔ ONNX equivalence
- **Quantize** to INT8 with `onnxruntime.quantization.quantize_static` (static calibration)
- **C++ inference** with ONNX Runtime + OpenCV: preprocess → infer → postprocess → draw at 30+ FPS
- **Accuracy validation**: mAP@50 drop < 2% after INT8 quantization

## Tech Stack

Python 3.11 · PyTorch · HuggingFace Transformers · ONNX Runtime · C++17 · OpenCV · CMake

## Quickstart

```bash
# Python (training + export)
cd training && pip install -r requirements.txt
python src/train.py          # fine-tune DETR
python src/export.py         # export to ONNX
python scripts/quantize.py   # INT8 quantization
python scripts/benchmark.py  # verify <33ms latency

# C++ inference
mkdir inference/build && cd inference/build
cmake .. -DCMAKE_TOOLCHAIN_FILE=../../vcpkg/scripts/buildsystems/vcpkg.cmake
make -j$(nproc)
./detector --model ../models/model_int8.onnx --source 0
```

## Architecture

```
[HuggingFace DETR] → fine-tune → checkpoints/best.pt
    → torch.onnx.export → model.onnx (FP32)
    → quantize_static  → model_int8.onnx (INT8)
    → C++ ONNX Runtime inference
        OpenCV capture → preprocess (resize+normalize) → OrtSession.Run
        → softmax decode → box coords → draw_detections → imshow @ 30+ FPS
```

## Project Structure

```
percept_stream/
├── training/
│   ├── src/            # dataset.py, train.py, evaluate.py, export.py
│   └── configs/        # detr_finetune.yaml
├── inference/
│   ├── src/            # detector.cpp, preprocess.cpp, postprocess.cpp, main.cpp
│   ├── tests/          # Google Test suite
│   └── CMakeLists.txt
└── scripts/            # quantize.py, benchmark.py
```
