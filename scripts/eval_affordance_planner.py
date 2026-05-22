#!/usr/bin/env python
import argparse
import sys
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

import numpy as np
import torch
import torchvision.transforms as T

from omnitry.enhance.data import (
    dice_score,
    image_exists,
    iou_score,
    load_json,
    load_person_image,
    weak_heatmap,
    write_json,
)
from omnitry.enhance.planner import class_to_id, load_planner


def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate the lightweight affordance planner on a manifest.")
    parser.add_argument("--manifest", default="data/hard_cases/omnitry_hard_cases.json")
    parser.add_argument("--checkpoint", default="checkpoints/enhance/affordance_planner.pt")
    parser.add_argument("--output", default="outputs/enhance/planner_eval.json")
    parser.add_argument("--image-size", type=int, default=64)
    parser.add_argument("--max-items", type=int, default=None)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    return parser.parse_args()


def main():
    args = parse_args()
    payload = load_json(Path(args.manifest))
    items = payload["items"] if isinstance(payload, dict) and "items" in payload else payload
    items = [item for item in items if image_exists(item)]
    if args.max_items is not None:
        items = items[: args.max_items]
    if not items:
        raise ValueError(f"No eval items with local images found in {args.manifest}")

    model, class_to_idx, meta = load_planner(args.checkpoint, map_location=args.device)
    model.to(args.device).eval()
    transform = T.ToTensor()

    results = []
    with torch.no_grad():
        for item in items:
            category = item.get("category") or item.get("garment_class", "")
            image = transform(load_person_image(item, args.image_size)).unsqueeze(0).to(args.device)
            class_id = torch.tensor([class_to_id(category, class_to_idx)], dtype=torch.long, device=args.device)
            pred = torch.sigmoid(model(image, class_id))[0, 0].cpu().numpy()
            target = weak_heatmap(category, args.image_size)
            results.append(
                {
                    "id": item.get("id"),
                    "category": category,
                    "dice": round(dice_score(pred, target), 6),
                    "iou": round(iou_score(pred, target), 6),
                    "pred_mean": round(float(pred.mean()), 6),
                    "target_mean": round(float(target.mean()), 6),
                }
            )

    summary = {
        "items": len(results),
        "dice_mean": round(float(np.mean([item["dice"] for item in results])), 6),
        "iou_mean": round(float(np.mean([item["iou"] for item in results])), 6),
    }
    write_json(
        Path(args.output),
        {
            "checkpoint": args.checkpoint,
            "checkpoint_meta": meta,
            "summary": summary,
            "items": results,
        },
    )
    print(summary)
    print(f"Wrote eval -> {args.output}")


if __name__ == "__main__":
    raise SystemExit(main())
