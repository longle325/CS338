#!/usr/bin/env python
import argparse
import random
import sys
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

import numpy as np
import torch
import torch.nn.functional as F
import torchvision.transforms as T
from torch.utils.data import DataLoader, Dataset

from omnitry.enhance.data import image_exists, load_person_image, weak_heatmap, write_json
from omnitry.enhance.planner import (
    AffordancePlanner,
    class_to_id,
    default_class_to_idx,
    save_planner,
)
from omnitry.enhance.data import load_json


class PlannerDataset(Dataset):
    def __init__(self, manifest_path, image_size=64, max_items=None):
        payload = load_json(Path(manifest_path))
        items = payload["items"] if isinstance(payload, dict) and "items" in payload else payload
        items = [item for item in items if image_exists(item)]
        if max_items is not None:
            items = items[:max_items]
        if not items:
            raise ValueError(f"No trainable items with local images found in {manifest_path}")
        self.items = items
        self.image_size = image_size
        self.transform = T.Compose([T.ToTensor()])
        self.class_to_idx = default_class_to_idx()

    def __len__(self):
        return len(self.items)

    def __getitem__(self, index):
        item = self.items[index]
        category = item.get("category") or item.get("garment_class", "")
        image = load_person_image(item, self.image_size)
        target = weak_heatmap(category, self.image_size)
        return {
            "image": self.transform(image),
            "target": torch.from_numpy(target).unsqueeze(0),
            "class_id": torch.tensor(class_to_id(category, self.class_to_idx), dtype=torch.long),
            "id": item.get("id", str(index)),
        }


def parse_args():
    parser = argparse.ArgumentParser(description="Train the lightweight OmniTry++ affordance planner.")
    parser.add_argument("--manifest", default="data/hard_cases/omnitry_hard_cases.json")
    parser.add_argument("--output", default="checkpoints/enhance/affordance_planner.pt")
    parser.add_argument("--metrics-output", default="outputs/enhance/planner_train_metrics.json")
    parser.add_argument("--image-size", type=int, default=64)
    parser.add_argument("--epochs", type=int, default=2)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--max-items", type=int, default=None)
    parser.add_argument("--seed", type=int, default=123)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    return parser.parse_args()


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def dice_from_logits(logits, target, threshold=0.5):
    pred = torch.sigmoid(logits) >= threshold
    truth = target >= threshold
    inter = (pred & truth).float().sum(dim=(1, 2, 3))
    denom = pred.float().sum(dim=(1, 2, 3)) + truth.float().sum(dim=(1, 2, 3))
    return torch.where(denom > 0, 2 * inter / denom, torch.ones_like(denom)).mean()


def main():
    args = parse_args()
    set_seed(args.seed)

    dataset = PlannerDataset(args.manifest, image_size=args.image_size, max_items=args.max_items)
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True, num_workers=0)
    model = AffordancePlanner(num_classes=len(dataset.class_to_idx)).to(args.device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.01)

    history = []
    model.train()
    for epoch in range(args.epochs):
        losses = []
        dices = []
        for batch in loader:
            image = batch["image"].to(args.device)
            target = batch["target"].to(args.device)
            class_id = batch["class_id"].to(args.device)
            logits = model(image, class_id)
            bce = F.binary_cross_entropy_with_logits(logits, target)
            pred = torch.sigmoid(logits)
            soft_inter = (pred * target).sum(dim=(1, 2, 3))
            soft_denom = pred.sum(dim=(1, 2, 3)) + target.sum(dim=(1, 2, 3)) + 1e-6
            dice_loss = 1 - (2 * soft_inter / soft_denom).mean()
            loss = bce + dice_loss

            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()

            losses.append(float(loss.detach().cpu()))
            dices.append(float(dice_from_logits(logits.detach(), target).cpu()))

        metrics = {
            "epoch": epoch + 1,
            "loss": round(float(np.mean(losses)), 6),
            "dice": round(float(np.mean(dices)), 6),
        }
        history.append(metrics)
        print(metrics)

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    save_planner(
        output,
        model.cpu(),
        dataset.class_to_idx,
        {
            "image_size": args.image_size,
            "manifest": args.manifest,
            "epochs": args.epochs,
            "items": len(dataset),
            "history": history,
        },
    )
    write_json(
        Path(args.metrics_output),
        {
            "checkpoint": str(output),
            "items": len(dataset),
            "history": history,
        },
    )
    print(f"Saved planner checkpoint -> {output}")


if __name__ == "__main__":
    raise SystemExit(main())
