import torch
import numpy as np
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval


def evaluate(model, loader, processor, device) -> dict:
    model.eval()
    results = []
    with torch.no_grad():
        for batch in loader:
            pv = batch["pixel_values"].to(device)
            pm = batch["pixel_mask"].to(device)
            outputs = model(pixel_values=pv, pixel_mask=pm)
            target_sizes = torch.stack([torch.tensor([800, 800])] * pv.size(0))
            preds = processor.post_process_object_detection(outputs, threshold=0.5, target_sizes=target_sizes)
            for pred, label in zip(preds, batch["labels"]):
                img_id = label["image_id"].item() if hasattr(label["image_id"], "item") else int(label["image_id"])
                for score, cls, box in zip(pred["scores"], pred["labels"], pred["boxes"]):
                    x1, y1, x2, y2 = box.tolist()
                    results.append({
                        "image_id": img_id,
                        "category_id": int(cls),
                        "bbox": [x1, y1, x2 - x1, y2 - y1],
                        "score": float(score),
                    })
    if not results:
        return {"mAP_50": 0.0, "mAP_50_95": 0.0}
    return {"mAP_50": 0.0, "mAP_50_95": 0.0}  # Full COCO eval requires dataset; stub here
