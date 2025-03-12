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

```mermaid
flowchart TB
    subgraph train["Offline: training/ and scripts/ (Python)"]
        ds["dataset.py<br/>COCO-style loader"]
        cfg["configs/detr_finetune.yaml"]
        tr["train.py<br/>build_model, AdamW, AMP scaler,<br/>train_one_epoch with grad clipping"]
        ck["checkpoints/best.pt"]
        ex["export.py<br/>torch.onnx.export opset 16, then validate_onnx"]
        onnx["models/model.onnx (FP32)"]
        qz["scripts/quantize.py<br/>quantize_static with a calibration reader"]
        int8["models/model_int8.onnx (INT8)"]
        ev["evaluate.py<br/>pycocotools mAP"]
        bench["scripts/benchmark.py<br/>p50 / p95 / p99 latency"]
    end

    subgraph infer["Online: inference/ (C++17, ONNX Runtime + OpenCV)"]
        cap["cv::VideoCapture<br/>--source 0 or a file path"]
        pre["preprocess_frame<br/>resize to 800, normalize, CHW tensor"]
        det["Detector<br/>Ort::Session, 4 intra-op threads"]
        post["postprocess<br/>softmax over 100 queries, cxcywh to xyxy,<br/>scaled back to frame size, --threshold"]
        draw["draw_detections + FPS overlay<br/>cv::imshow, quit on q"]
    end

    ds --> tr
    cfg --> tr
    tr --> ck --> ex --> onnx --> qz --> int8
    onnx --> ev
    int8 --> ev
    int8 --> bench
    int8 -->|"--model"| det
    cap --> pre --> det --> post --> draw
    draw -->|"next frame"| cap
```

### Per-frame path

<img src="docs/frame-pipeline.svg" alt="Frames moving through capture, preprocess, ONNX Runtime inference, postprocess and draw" width="940">

## Evaluation

The model is evaluated on the COCO val split (or the held-out split of the
fine-tuning dataset) using standard detection metrics.

### Detection metrics

Computed by `training/src/evaluate.py` via `pycocotools` over the validation
split listed in `training/configs/detr_finetune.yaml`:

- **mAP@[0.5:0.95]** — primary COCO metric, IoU averaged 0.5..0.95 step 0.05
- **mAP@0.5** — single-IoU mAP (lenient localization)
- **mAP@0.75** — strict-localization mAP
- **AP per class** and **AR@1 / AR@10 / AR@100**
- **Precision-recall curves** per class (saved as PNGs)

Run evaluation on a checkpoint:

```bash
python training/src/evaluate.py \
  --checkpoint training/checkpoints/best.pt \
  --config training/configs/detr_finetune.yaml \
  --split val
```

### Quantization quality gate

After INT8 export, compare FP32 vs INT8 on the same val split. The pass
criterion is **mAP@0.5 drop < 2 percentage points** and **mAP@[0.5:0.95]
drop < 1 pp**:

```bash
python training/src/evaluate.py --checkpoint models/model.onnx       --runtime ort
python training/src/evaluate.py --checkpoint models/model_int8.onnx  --runtime ort
```

### Runtime / latency

`scripts/benchmark.py` measures end-to-end inference latency on a fixed
batch of images using the ONNX Runtime session, reporting:

- **p50 / p95 / p99 latency** (ms per frame)
- **mean FPS** under sustained load
- **Throughput (images/sec)** at batch size 1 and 8

```bash
python scripts/benchmark.py --model models/model_int8.onnx --num-iters 500
```

Target: **>=30 FPS** end-to-end (preprocess + infer + postprocess) on the
deployment GPU/CPU; p99 < 33 ms.

### C++ inference correctness

`inference/tests/test_detector.cpp` (Google Test) checks preprocess/postprocess
parity with the Python pipeline on a small fixture set:

```bash
cd inference/build && ctest --output-on-failure
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
