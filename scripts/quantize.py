"""INT8 static quantization via ONNX Runtime."""
import numpy as np
from onnxruntime.quantization import quantize_static, CalibrationDataReader, QuantType, QuantFormat


class DETRCalibrationReader(CalibrationDataReader):
    def __init__(self, num_samples: int = 100):
        self.num_samples = num_samples
        self._count = 0

    def get_next(self):
        if self._count >= self.num_samples:
            return None
        self._count += 1
        return {
            "pixel_values": np.random.randn(1, 3, 800, 800).astype(np.float32),
            "pixel_mask": np.ones((1, 800, 800), dtype=np.int64),
        }


def main():
    quantize_static(
        model_input="inference/models/model.onnx",
        model_output="inference/models/model_int8.onnx",
        calibration_data_reader=DETRCalibrationReader(100),
        quant_format=QuantFormat.QOperator,
        weight_type=QuantType.QInt8,
    )
    print("Quantized model saved to inference/models/model_int8.onnx")


if __name__ == "__main__":
    main()
