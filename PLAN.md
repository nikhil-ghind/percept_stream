# Percept Stream

## Project Overview

Fine-tune a DETR (Detection Transformer) model for a target domain and build a high-performance, production-grade inference pipeline capable of sustained 30+ FPS throughput. The model is exported to ONNX and compiled with the C++ ONNX Runtime, then optimized to INT8 precision via post-training quantization. An OpenCV-based capture-and-display loop ties the components together, making the pipeline suitable for deployment on edge servers or workstations without Python overhead at inference time.

Key goals:
- Fine-tune DETR on a domain-specific dataset (e.g. industrial parts, traffic objects).
- Export and quantize the model to ONNX INT8 achieving <33ms per frame (30+ FPS).
- Build a self-contained C++ inference application using ONNX Runtime + OpenCV.
- Validate detection accuracy (mAP@0.5) stays within 2% of FP32 baseline after quantization.

---

## Tech Stack

| Layer | Technology |
|---|---|
| Training | PyTorch 2.2, torchvision 0.17, transformers 4.39 (HuggingFace DETR) |
| Export / quantization | torch.onnx, onnxruntime 1.17, onnxruntime-tools, onnx 1.15 |
| Inference runtime | ONNX Runtime C++ API 1.17 |
| Video I/O (C++) | OpenCV 4.9 |
| Build system | CMake 3.27, vcpkg or Conan for C++ deps |
| Evaluation | pycocotools, numpy, matplotlib |
| Testing | pytest (Python), Google Test (C++) |
| Environment | Python 3.11, C++17, Ubuntu 22.04 |

---

## Architecture Overview

```
[Training: Python]
  HuggingFace DETR --> fine-tune on custom COCO-format dataset
        |
        v
  Export: torch.onnx.export --> model.onnx (FP32)
        |
        v
  Quantize: onnxruntime-tools --> model_int8.onnx (INT8)
        |
        v
[Inference: C++]
  OpenCV VideoCapture --> frame --> pre-process (resize, normalize)
        |
        v
  Ort::Session.Run(model_int8.onnx)
        |
        v
  Post-process (NMS, decode boxes) --> draw bboxes --> imshow / write
```

---

## Phase 1: Project Layout and Environment Setup

**Goal**: Establish the directory structure and install all Python and C++ toolchains.

### Tasks

1. Create the project directory layout:
   ```
   realtimeObjectDetectionPipeline/
   ├── training/
   │   ├── data/
   │   ├── configs/
   │   │   └── detr_finetune.yaml
   │   ├── src/
   │   │   ├── __init__.py
   │   │   ├── dataset.py
   │   │   ├── train.py
   │   │   ├── evaluate.py
   │   │   └── export.py
   │   ├── checkpoints/
   │   └── requirements.txt
   ├── inference/
   │   ├── CMakeLists.txt
   │   ├── src/
   │   │   ├── main.cpp
   │   │   ├── detector.hpp
   │   │   ├── detector.cpp
   │   │   ├── preprocess.hpp
   │   │   ├── preprocess.cpp
   │   │   ├── postprocess.hpp
   │   │   └── postprocess.cpp
   │   ├── tests/
   │   │   ├── CMakeLists.txt
   │   │   └── test_detector.cpp
   │   └── models/
   └── scripts/
       ├── quantize.py
       └── benchmark.py
   ```

2. Create `training/requirements.txt`:
   ```
   torch==2.2.2
   torchvision==0.17.2
   transformers==4.39.3
   datasets==2.18.0
   onnx==1.15.0
   onnxruntime==1.17.1
   onnxruntime-tools==1.7.0
   pycocotools==2.0.7
   numpy==1.26.4
   Pillow==10.2.0
   matplotlib==3.8.3
   tqdm==4.66.2
   pytest==8.1.1
   pyyaml==6.0.1
   ```

3. Install Python dependencies:
   ```
   python -m venv .venv && source .venv/bin/activate
   pip install -r training/requirements.txt
   ```

4. Install C++ dependencies using vcpkg:
   ```
   git clone https://github.com/microsoft/vcpkg.git
   ./vcpkg/bootstrap-vcpkg.sh
   ./vcpkg/vcpkg install onnxruntime opencv4
   ```

5. Create `training/configs/detr_finetune.yaml`:
   ```yaml
   model:
     name: facebook/detr-resnet-50
     num_labels: <N>   # set to actual number of classes
   training:
     epochs: 50
     batch_size: 8
     lr: 1e-5
     lr_backbone: 1e-6
     weight_decay: 1e-4
     warmup_steps: 500
     gradient_clip: 0.1
   data:
     train_dir: data/train
     val_dir: data/val
     annotation_file_train: data/annotations/train.json
     annotation_file_val: data/annotations/val.json
     image_size: 800
   output:
     checkpoint_dir: checkpoints/
     export_path: ../inference/models/model.onnx
   ```

---

## Phase 2: Dataset Preparation

**Goal**: Load and preprocess a COCO-format dataset for DETR fine-tuning.

### Tasks

1. Create `training/src/dataset.py` — class `CocoDetectionDataset(torch.utils.data.Dataset)`:
   - `__init__(self, img_dir: str, annotation_file: str, processor)`: loads COCO annotations using `pycocotools.coco.COCO`, stores image paths and annotations, stores HuggingFace `DetrImageProcessor`.
   - `__len__`: returns number of images.
   - `__getitem__(self, idx)`: loads image with `PIL.Image.open`, builds `annotations` dict in HuggingFace format `{"image_id": int, "annotations": [{"bbox": [...], "category_id": int, ...}]}`, runs `self.processor(images=image, annotations=annotations, return_tensors="pt")`, squeezes batch dim. Returns `(pixel_values, pixel_mask, labels)`.
   - `collate_fn(batch)`: pads pixel_values/pixel_mask to same size in batch; returns dict with `pixel_values`, `pixel_mask`, `labels`.

2. Create `training/data/download_sample.py` (or describe manual step): download a small subset (500 images) from COCO 2017 val set, filter to 5 categories, write to `data/` in COCO JSON format.

3. Verify dataset with a quick sanity check:
   ```
   python -c "from src.dataset import CocoDetectionDataset; ..."
   ```
   Print one batch shape and confirm it is `(B, 3, H, W)`.

---

## Phase 3: DETR Fine-Tuning

**Goal**: Fine-tune `facebook/detr-resnet-50` on the target dataset and save the best checkpoint.

### Tasks

1. Create `training/src/train.py`:
   - `build_model(config: dict) -> DetrForObjectDetection`: loads `DetrForObjectDetection.from_pretrained(config["model"]["name"], num_labels=config["model"]["num_labels"], ignore_mismatched_sizes=True)`.
   - `build_optimizer(model, config) -> AdamW`: sets two param groups — backbone params with `lr_backbone`, all other params with `lr`. Applies `weight_decay`.
   - `build_scheduler(optimizer, config) -> LinearWarmupCosineAnnealingLR`: use `transformers.get_cosine_schedule_with_warmup(optimizer, num_warmup_steps, num_training_steps)`.
   - `train_one_epoch(model, loader, optimizer, scheduler, scaler, device, grad_clip) -> float`: iterates loader, calls `model(**batch)`, takes `loss = outputs.loss`, backward, grad clip, optimizer step, scheduler step. Returns mean loss.
   - `main()`:
     - Load config from `configs/detr_finetune.yaml`.
     - Build datasets, dataloaders (num_workers=4).
     - Build model, optimizer, scheduler.
     - Use `torch.cuda.amp.GradScaler` for mixed precision.
     - Training loop for `epochs`: call `train_one_epoch`, call `evaluate` from `evaluate.py`, save checkpoint if mAP improves.
     - Save best checkpoint to `checkpoints/best.pt`.

2. Create `training/src/evaluate.py`:
   - `evaluate(model, loader, processor, device) -> dict`: runs inference on val set, converts predictions to COCO format, runs `pycocotools` evaluator. Returns `{"mAP_50": float, "mAP_50_95": float}`.
   - `main()`: load checkpoint, run evaluate, print results.

3. Run training (minimum 10 epochs on sample data to confirm pipeline):
   ```
   python src/train.py
   ```

---

## Phase 4: ONNX Export and INT8 Quantization

**Goal**: Export the fine-tuned model to ONNX FP32, validate output equivalence, then quantize to INT8.

### Tasks

1. Create `training/src/export.py`:
   - `load_model_from_checkpoint(checkpoint_path: str, config: dict) -> DetrForObjectDetection`: loads state dict, sets `model.eval()`.
   - `export_to_onnx(model, processor, output_path: str, opset: int = 16)`:
     - Create a dummy input: `pixel_values = torch.randn(1, 3, 800, 800)`, `pixel_mask = torch.ones(1, 800, 800, dtype=torch.long)`.
     - Call `torch.onnx.export(model, (pixel_values, pixel_mask), output_path, opset_version=opset, input_names=["pixel_values", "pixel_mask"], output_names=["logits", "pred_boxes"], dynamic_axes={"pixel_values": {0: "batch"}, "pixel_mask": {0: "batch"}, "logits": {0: "batch"}, "pred_boxes": {0: "batch"}})`.
     - Verify with `onnx.checker.check_model(onnx.load(output_path))`.
   - `validate_onnx(model, onnx_path: str, processor, sample_image_path: str)`:
     - Run PyTorch model and ONNX Runtime on same input.
     - Assert `np.allclose(pt_logits, ort_logits, atol=1e-4)`.
   - `main()`: export and validate.
   - Run: `python src/export.py`

2. Create `scripts/quantize.py`:
   - Use `onnxruntime.quantization.quantize_static` with `ActivationSymmetric=True`, `weight_type=QuantType.QInt8`.
   - Define `CalibrationDataReader` class that yields preprocessed batches from `data/val/` (100 images).
   - Call `quantize_static(input_model_path="models/model.onnx", output_model_path="models/model_int8.onnx", calibration_data_reader=reader, quant_format=QuantFormat.QOperator)`.
   - Run: `python scripts/quantize.py`

3. Create `scripts/benchmark.py`:
   - Load `model_int8.onnx` via `onnxruntime.InferenceSession`.
   - Run inference on 200 random inputs of shape `(1, 3, 800, 800)`.
   - Compute mean, p50, p95, p99 latency. Print results. Assert mean < 33ms.

---

## Phase 5: C++ Inference Application

**Goal**: Implement a self-contained C++ application that reads frames from a webcam or video file, runs INT8 ONNX inference, and overlays detections at 30+ FPS.

### Tasks

1. Create `inference/CMakeLists.txt`:
   ```cmake
   cmake_minimum_required(VERSION 3.27)
   project(ObjectDetectionPipeline CXX)
   set(CMAKE_CXX_STANDARD 17)
   find_package(onnxruntime REQUIRED)
   find_package(OpenCV REQUIRED)
   add_executable(detector src/main.cpp src/detector.cpp src/preprocess.cpp src/postprocess.cpp)
   target_link_libraries(detector PRIVATE onnxruntime::onnxruntime ${OpenCV_LIBS})
   add_subdirectory(tests)
   ```

2. Create `inference/src/preprocess.hpp` and `preprocess.cpp`:
   - `struct PreprocessedInput { std::vector<float> pixel_values; std::vector<int64_t> pixel_mask; std::array<int64_t, 4> shape; };`
   - `PreprocessedInput preprocess_frame(const cv::Mat& frame, int target_size = 800)`:
     - Resize frame to `target_size x target_size` with `cv::INTER_LINEAR`.
     - Convert BGR to RGB.
     - Normalize with ImageNet mean `{0.485, 0.456, 0.406}` and std `{0.229, 0.224, 0.225}`.
     - Convert HWC to CHW layout.
     - Fill `pixel_mask` with all-ones of shape `(1, target_size, target_size)`.
     - Return `PreprocessedInput` with flat float vector and shape `{1, 3, target_size, target_size}`.

3. Create `inference/src/postprocess.hpp` and `postprocess.cpp`:
   - `struct Detection { int class_id; float confidence; cv::Rect2f box; };`
   - `std::vector<Detection> postprocess(const float* logits, const float* pred_boxes, int num_queries, int num_classes, float score_threshold = 0.5f, int img_w = 800, int img_h = 800)`:
     - For each query: softmax logits → `class_id = argmax`, `confidence = max_prob` (excluding no-object class).
     - Filter by `confidence > score_threshold`.
     - Convert center-format boxes `(cx, cy, w, h)` in `[0,1]` to pixel `cv::Rect2f` scaled by `(img_w, img_h)`.
     - Return filtered list.
   - `void draw_detections(cv::Mat& frame, const std::vector<Detection>& dets, const std::vector<std::string>& class_names)`: draws bounding boxes and labels using `cv::rectangle` and `cv::putText`.

4. Create `inference/src/detector.hpp` and `detector.cpp`:
   - `class Detector`:
     - Private members: `Ort::Env env_`, `Ort::Session session_`, `Ort::SessionOptions opts_`, `std::vector<std::string> input_names_`, `std::vector<std::string> output_names_`.
     - `Detector(const std::string& model_path, int num_threads = 4)`: construct `Ort::Env(ORT_LOGGING_LEVEL_WARNING, "detector")`, set `opts_.SetIntraOpNumThreads(num_threads)`, set `opts_.SetGraphOptimizationLevel(ORT_ENABLE_ALL)`, create `Ort::Session(env_, model_path.c_str(), opts_)`.
     - `std::pair<std::vector<float>, std::vector<float>> run(const PreprocessedInput& input)`: creates `Ort::MemoryInfo`, builds `Ort::Value` tensors for pixel_values and pixel_mask, calls `session_.Run(...)`, copies output tensors to float vectors. Returns `(logits, pred_boxes)`.

5. Create `inference/src/main.cpp`:
   - Parse CLI args: `--model` (path to `model_int8.onnx`), `--source` (video file path or `0` for webcam), `--labels` (path to labels text file), `--threshold` (float, default 0.5).
   - Open `cv::VideoCapture` with source.
   - Instantiate `Detector`.
   - Frame loop:
     - `cap.read(frame)`.
     - `auto t0 = std::chrono::high_resolution_clock::now()`.
     - `auto input = preprocess_frame(frame)`.
     - `auto [logits, boxes] = detector.run(input)`.
     - `auto dets = postprocess(logits.data(), boxes.data(), 100, num_classes, threshold, frame.cols, frame.rows)`.
     - `draw_detections(frame, dets, class_names)`.
     - Compute FPS, overlay with `cv::putText`.
     - `cv::imshow("Detections", frame)`.
   - Press `q` to exit.

6. Create `inference/tests/test_detector.cpp` using Google Test:
   - Test `preprocess_frame`: verify output shape is `(1, 3, 800, 800)`, values in `[-3, 3]`.
   - Test `postprocess`: construct synthetic logits/boxes with one high-confidence detection, assert one `Detection` is returned with correct class.
   - Test `Detector::run`: load `model_int8.onnx`, run with dummy input, assert output sizes are `(1, 100, num_classes+1)` and `(1, 100, 4)`.

7. Build and run:
   ```
   mkdir inference/build && cd inference/build
   cmake .. -DCMAKE_TOOLCHAIN_FILE=../../vcpkg/scripts/buildsystems/vcpkg.cmake
   make -j$(nproc)
   ./detector --model ../models/model_int8.onnx --source 0 --labels ../models/labels.txt
   ```

8. Verify FPS overlay shows >= 30 FPS on a 1080p input stream.

---

## Phase 6: Accuracy Validation and Benchmarking

**Goal**: Confirm INT8 quantization does not degrade mAP by more than 2% relative to FP32, and document latency results.

### Tasks

1. Create `scripts/eval_onnx.py`:
   - Load `model.onnx` (FP32) and `model_int8.onnx` (INT8).
   - Run both on full val set using ONNX Runtime Python API.
   - Evaluate mAP using pycocotools for both.
   - Print comparison table.
   - Assert `mAP_fp32 - mAP_int8 < 0.02`.

2. Run the C++ benchmark via `benchmark.py` and record results to `results/latency_report.txt`.

3. Run Google Test suite:
   ```
   cd inference/build && ctest --output-on-failure
   ```
   All tests must pass.

4. Run Python test suite:
   ```
   pytest training/ -v
   ```
   All tests must pass.
