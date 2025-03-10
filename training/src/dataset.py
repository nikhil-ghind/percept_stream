import torch
from torch.utils.data import Dataset
from PIL import Image
from pycocotools.coco import COCO


class CocoDetectionDataset(Dataset):
    def __init__(self, img_dir: str, annotation_file: str, processor):
        self.img_dir = img_dir
        self.coco = COCO(annotation_file)
        self.ids = list(self.coco.imgs.keys())
        self.processor = processor

    def __len__(self):
        return len(self.ids)

    def __getitem__(self, idx):
        img_id = self.ids[idx]
        img_info = self.coco.imgs[img_id]
        img = Image.open(f"{self.img_dir}/{img_info['file_name']}").convert("RGB")
        ann_ids = self.coco.getAnnIds(imgIds=img_id)
        anns = self.coco.loadAnns(ann_ids)
        annotations = {"image_id": img_id, "annotations": anns}
        encoding = self.processor(images=img, annotations=annotations, return_tensors="pt")
        return {k: v.squeeze(0) for k, v in encoding.items()}

    @staticmethod
    def collate_fn(batch):
        pixel_values = torch.stack([b["pixel_values"] for b in batch])
        pixel_mask = torch.stack([b["pixel_mask"] for b in batch])
        labels = [b["labels"] for b in batch]
        return {"pixel_values": pixel_values, "pixel_mask": pixel_mask, "labels": labels}
