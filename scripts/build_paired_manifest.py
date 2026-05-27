#!/usr/bin/env python
import argparse
import json
import sys
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

import numpy as np
from PIL import Image, ImageFilter

from omnitry.enhance.data import hardness_score, select_hard_cases, summarize_manifest, write_json
from omnitry.enhance.taxonomy import normalize_class_name


def parse_args():
    parser = argparse.ArgumentParser(description="Build paired or pseudo-paired OmniTry fine-tuning manifests.")
    parser.add_argument("--bench-root", default="data/OmniTry_Bench")
    parser.add_argument("--index", default="data/OmniTry_Bench/omni_vtryon_bench_v1.json")
    parser.add_argument("--output", default="data/hard_cases/omnitry_pseudo_paired_train.json")
    parser.add_argument("--asset-dir", default="data/pseudo_pairs/omnitry_bench")
    parser.add_argument("--top-k", type=int, default=1000)
    parser.add_argument("--per-class", type=int, default=80)
    parser.add_argument("--max-items", type=int, default=None)
    parser.add_argument("--min-mask-pixels", type=int, default=64)
    parser.add_argument("--crop-margin", type=float, default=0.20)
    return parser.parse_args()


def load_json(path):
    with Path(path).open("r", encoding="utf-8") as handle:
        return json.load(handle)


def resolve_bench_path(bench_root, raw_path):
    bench_root = Path(bench_root)
    raw = Path(raw_path)
    if raw.is_absolute():
        return raw
    direct = bench_root / raw
    if direct.exists():
        return direct
    if raw.parts and raw.parts[0] == bench_root.name:
        return bench_root.joinpath(*raw.parts[1:])
    return direct


def mask_path_for(image_path):
    return image_path.with_name(f"{image_path.stem}_mask{image_path.suffix}")


def mask_bbox(mask, threshold=8):
    arr = np.asarray(mask.convert("L"))
    ys, xs = np.nonzero(arr > threshold)
    if xs.size == 0:
        return None, 0
    return (int(xs.min()), int(ys.min()), int(xs.max()) + 1, int(ys.max()) + 1), int(xs.size)


def expand_box(box, width, height, margin):
    x1, y1, x2, y2 = box
    bw = x2 - x1
    bh = y2 - y1
    pad_x = int(bw * margin)
    pad_y = int(bh * margin)
    return (
        max(0, x1 - pad_x),
        max(0, y1 - pad_y),
        min(width, x2 + pad_x),
        min(height, y2 + pad_y),
    )


def erase_masked_region(image, mask):
    image = image.convert("RGB")
    mask = mask.convert("L").resize(image.size, Image.Resampling.NEAREST)
    alpha = mask.filter(ImageFilter.GaussianBlur(radius=5))
    blurred = image.filter(ImageFilter.GaussianBlur(radius=18))
    white = Image.new("RGB", image.size, (255, 255, 255))
    fill = Image.blend(blurred, white, 0.55)
    return Image.composite(fill, image, alpha)


def object_crop_from_mask(image, mask, box):
    image = image.convert("RGB")
    mask = mask.convert("L").resize(image.size, Image.Resampling.NEAREST)
    crop = image.crop(box)
    crop_mask = mask.crop(box)
    white = Image.new("RGB", crop.size, (255, 255, 255))
    return Image.composite(crop, white, crop_mask)


def build_item(raw_item, bench_root, asset_dir, args):
    person_path = resolve_bench_path(bench_root, raw_item.get("person", {}).get("img_path", ""))
    if not person_path.is_file():
        return None
    person_mask_path = mask_path_for(person_path)
    if not person_mask_path.is_file():
        return None

    image = Image.open(person_path).convert("RGB")
    mask = Image.open(person_mask_path).convert("L")
    if mask.size != image.size:
        mask = mask.resize(image.size, Image.Resampling.NEAREST)
    box, mask_pixels = mask_bbox(mask)
    if box is None or mask_pixels < args.min_mask_pixels:
        return None

    item_id = raw_item.get("id", person_path.stem)
    category = normalize_class_name(raw_item.get("garment_class") or raw_item.get("class_name", "").split("_")[0])
    item_dir = Path(asset_dir) / category.replace(" ", "_") / item_id
    item_dir.mkdir(parents=True, exist_ok=True)

    source_path = item_dir / "person_erased.jpg"
    object_path = item_dir / "object_crop.jpg"
    target_path = item_dir / "target.jpg"

    if not source_path.is_file():
        erase_masked_region(image, mask).save(source_path, quality=95)
    if not object_path.is_file():
        crop_box = expand_box(box, image.width, image.height, args.crop_margin)
        object_crop_from_mask(image, mask, crop_box).save(object_path, quality=95)
    if not target_path.is_file():
        image.save(target_path, quality=95)

    copied = dict(raw_item)
    copied["id"] = item_id
    copied["source"] = "omnitry_bench_pseudo_pair"
    copied["pairing"] = "person_mask_self_reconstruction"
    copied["category"] = category
    copied["person_path"] = str(source_path)
    copied["object_path"] = str(object_path)
    copied["target_path"] = str(target_path)
    copied["original_person_path"] = str(person_path)
    copied["person_mask_path"] = str(person_mask_path)
    copied["hardness_score"] = hardness_score(copied)
    return copied


def category_for(raw_item):
    return normalize_class_name(raw_item.get("garment_class") or raw_item.get("class_name", "").split("_")[0])


def main():
    args = parse_args()
    bench_root = Path(args.bench_root)
    raw_items = load_json(args.index)
    if args.max_items is not None:
        raw_items = raw_items[: args.max_items]

    ranked_raw = []
    for raw_item in raw_items:
        copied = dict(raw_item)
        copied["category"] = category_for(copied)
        copied["hardness_score"] = hardness_score(copied)
        ranked_raw.append(copied)
    ranked_raw = sorted(ranked_raw, key=lambda item: item.get("hardness_score", 0.0), reverse=True)

    items = []
    class_counts = {}
    skipped = 0
    for raw_item in ranked_raw:
        category = raw_item["category"]
        if args.per_class > 0 and class_counts.get(category, 0) >= args.per_class:
            continue
        item = build_item(raw_item, bench_root, Path(args.asset_dir), args)
        if item is None:
            skipped += 1
            continue
        items.append(item)
        class_counts[category] = class_counts.get(category, 0) + 1
        if len(items) % 100 == 0:
            print(f"Built {len(items)} pseudo-paired items...")
        if len(items) >= args.top_k:
            break

    selected = select_hard_cases(items, top_k=args.top_k, per_class=args.per_class)
    payload = {
        "source": "omnitry_bench_pseudo_pair",
        "pairing": "person_mask_self_reconstruction",
        "summary": summarize_manifest(selected),
        "skipped": skipped,
        "items": selected,
    }
    write_json(Path(args.output), payload)
    print(f"Wrote {len(selected)} pseudo-paired items -> {args.output}")
    print(payload["summary"])
    print(f"Skipped items without usable person masks: {skipped}")


if __name__ == "__main__":
    raise SystemExit(main())
