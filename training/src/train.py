import yaml
import torch
from torch.utils.data import DataLoader
from transformers import DetrForObjectDetection, DetrImageProcessor, get_cosine_schedule_with_warmup
from torch.cuda.amp import GradScaler

from src.dataset import CocoDetectionDataset
from src.evaluate import evaluate


def build_model(config: dict):
    return DetrForObjectDetection.from_pretrained(
        config["model"]["name"],
        num_labels=config["model"]["num_labels"],
        ignore_mismatched_sizes=True,
    )


def build_optimizer(model, config: dict):
    backbone_params = [p for n, p in model.named_parameters() if "backbone" in n]
    other_params = [p for n, p in model.named_parameters() if "backbone" not in n]
    return torch.optim.AdamW([
        {"params": backbone_params, "lr": config["training"]["lr_backbone"]},
        {"params": other_params, "lr": config["training"]["lr"]},
    ], weight_decay=config["training"]["weight_decay"])


def train_one_epoch(model, loader, optimizer, scheduler, scaler, device, grad_clip) -> float:
    model.train()
    total_loss = 0.0
    for batch in loader:
        pixel_values = batch["pixel_values"].to(device)
        pixel_mask = batch["pixel_mask"].to(device)
        labels = [{k: v.to(device) for k, v in t.items()} for t in batch["labels"]]

        optimizer.zero_grad()
        with torch.cuda.amp.autocast(enabled=scaler.is_enabled()):
            outputs = model(pixel_values=pixel_values, pixel_mask=pixel_mask, labels=labels)
            loss = outputs.loss

        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
        scaler.step(optimizer)
        scaler.update()
        scheduler.step()
        total_loss += loss.item()

    return total_loss / len(loader)


def main():
    with open("configs/detr_finetune.yaml") as f:
        config = yaml.safe_load(f)

    processor = DetrImageProcessor.from_pretrained(config["model"]["name"])
    train_ds = CocoDetectionDataset(config["data"]["train_dir"], config["data"]["annotation_file_train"], processor)
    val_ds = CocoDetectionDataset(config["data"]["val_dir"], config["data"]["annotation_file_val"], processor)

    train_loader = DataLoader(train_ds, batch_size=config["training"]["batch_size"],
                              shuffle=True, collate_fn=CocoDetectionDataset.collate_fn, num_workers=4)
    val_loader = DataLoader(val_ds, batch_size=config["training"]["batch_size"],
                            collate_fn=CocoDetectionDataset.collate_fn, num_workers=4)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = build_model(config).to(device)
    optimizer = build_optimizer(model, config)

    total_steps = config["training"]["epochs"] * len(train_loader)
    scheduler = get_cosine_schedule_with_warmup(optimizer, config["training"]["warmup_steps"], total_steps)
    scaler = GradScaler()

    best_map = 0.0
    for epoch in range(1, config["training"]["epochs"] + 1):
        loss = train_one_epoch(model, train_loader, optimizer, scheduler, scaler, device, config["training"]["gradient_clip"])
        metrics = evaluate(model, val_loader, processor, device)
        print(f"Epoch {epoch} | loss={loss:.4f} | mAP_50={metrics.get('mAP_50', 0):.4f}")
        if metrics.get("mAP_50", 0) > best_map:
            best_map = metrics["mAP_50"]
            torch.save(model.state_dict(), f"{config['output']['checkpoint_dir']}/best.pt")


if __name__ == "__main__":
    main()
