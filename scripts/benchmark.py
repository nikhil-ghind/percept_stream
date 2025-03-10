"""Benchmark INT8 ONNX model latency."""
import time
import numpy as np
import onnxruntime as ort


def main():
    sess = ort.InferenceSession("inference/models/model_int8.onnx")
    latencies = []
    for _ in range(200):
        x = np.random.randn(1, 3, 800, 800).astype(np.float32)
        m = np.ones((1, 800, 800), dtype=np.int64)
        t0 = time.perf_counter()
        sess.run(None, {"pixel_values": x, "pixel_mask": m})
        latencies.append((time.perf_counter() - t0) * 1000)

    latencies.sort()
    mean_ms = np.mean(latencies)
    p50 = np.percentile(latencies, 50)
    p95 = np.percentile(latencies, 95)
    p99 = np.percentile(latencies, 99)
    print(f"mean={mean_ms:.1f}ms  p50={p50:.1f}ms  p95={p95:.1f}ms  p99={p99:.1f}ms")
    assert mean_ms < 33.0, f"Mean latency {mean_ms:.1f}ms exceeds 33ms target"
    print("Latency target met.")


if __name__ == "__main__":
    main()
