import yaml
import torch
import onnx
import onnxruntime as ort
import numpy as np
from transformers import DetrForObjectDetection, DetrImageProcessor

from src.train import build_model


def load_model_from_checkpoint(checkpoint_path: str, config: dict):
    model = build_model(config)
    model.load_state_dict(torch.load(checkpoint_path, map_location="cpu"))
    model.eval()
    return model


def export_to_onnx(model, output_path: str, opset: int = 16):
    pixel_values = torch.randn(1, 3, 800, 800)
    pixel_mask = torch.ones(1, 800, 800, dtype=torch.long)
    torch.onnx.export(
        model,
        (pixel_values, pixel_mask),
        output_path,
        opset_version=opset,
        input_names=["pixel_values", "pixel_mask"],
        output_names=["logits", "pred_boxes"],
        dynamic_axes={
            "pixel_values": {0: "batch"},
            "pixel_mask": {0: "batch"},
            "logits": {0: "batch"},
            "pred_boxes": {0: "batch"},
        },
    )
    onnx.checker.check_model(onnx.load(output_path))
    print(f"ONNX model exported to {output_path}")


def validate_onnx(model, onnx_path: str):
    x = torch.randn(1, 3, 800, 800)
    m = torch.ones(1, 800, 800, dtype=torch.long)
    with torch.no_grad():
        pt_out = model(pixel_values=x, pixel_mask=m)
        pt_logits = pt_out.logits.numpy()

    sess = ort.InferenceSession(onnx_path)
    ort_out = sess.run(None, {"pixel_values": x.numpy(), "pixel_mask": m.numpy()})
    assert np.allclose(pt_logits, ort_out[0], atol=1e-3), "ONNX/PyTorch output mismatch"
    print("ONNX validation passed.")


if __name__ == "__main__":
    with open("configs/detr_finetune.yaml") as f:
        config = yaml.safe_load(f)
    model = load_model_from_checkpoint(f"{config['output']['checkpoint_dir']}/best.pt", config)
    export_to_onnx(model, config["output"]["export_path"])
    validate_onnx(model, config["output"]["export_path"])
