#!/usr/bin/env python
import argparse
import json
import re
import sys
import urllib.parse
from pathlib import Path

import numpy as np
import requests
from PIL import Image, ImageDraw, ImageFilter, ImageFile

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from omnitry.enhance.data import hardness_score, summarize_manifest, write_json
from omnitry.enhance.taxonomy import normalize_class_name

ImageFile.LOAD_TRUNCATED_IMAGES = True


def parse_args():
    parser = argparse.ArgumentParser(description="Build pseudo-paired training data from LLM-labeled Commons images.")
    parser.add_argument("--input", default="data/hard_cases/commons_llm_usable_labels.json")
    parser.add_argument("--output", default="data/hard_cases/commons_pseudo_paired_train.json")
    parser.add_argument("--asset-dir", default="data/pseudo_pairs/commons_llm")
    parser.add_argument("--crop-margin", type=float, default=0.25)
    parser.add_argument("--min-mask-pixels", type=int, default=64)
    parser.add_argument("--max-items", type=int, default=None)
    return parser.parse_args()


def load_json(path):
    with Path(path).open("r", encoding="utf-8") as handle:
        return json.load(handle)


def slug(value):
    value = re.sub(r"[^a-zA-Z0-9._-]+", "_", str(value).strip())
    value = value.strip("._")
    return value[:120] or "item"


def resolve_local(path):
    if not path:
        return None
    path = Path(str(path).replace("file://", ""))
    if path.is_absolute():
        return path
    return ROOT_DIR / path


def download_image(item, raw_dir):
    local_path = resolve_local(item.get("local_path"))
    if local_path and local_path.is_file():
        return local_path

    url = item.get("thumb_url") or item.get("url")
    if not url:
        return None
    suffix = Path(urllib.parse.urlparse(url).path).suffix or ".jpg"
    if len(suffix) > 8:
        suffix = ".jpg"
    category = normalize_class_name(item.get("category") or "unknown").replace(" ", "_")
    output = raw_dir / category / f"{slug(item.get('id'))}{suffix}"
    output.parent.mkdir(parents=True, exist_ok=True)
    if output.is_file():
        return output

    response = requests.get(
        url,
        headers={"User-Agent": "OmniTry-CS338-pseudo-pair-builder/0.1 (educational research)"},
        timeout=45,
    )
    response.raise_for_status()
    output.write_bytes(response.content)
    return output


def bbox_to_pixels(bbox, width, height):
    x1, y1, x2, y2 = [float(v) for v in bbox]
    if x2 < x1:
        x1, x2 = x2, x1
    if y2 < y1:
        y1, y2 = y2, y1
    return (
        max(0, min(width - 1, int(round(x1 * width)))),
        max(0, min(height - 1, int(round(y1 * height)))),
        max(1, min(width, int(round(x2 * width)))),
        max(1, min(height, int(round(y2 * height)))),
    )


def expand_box(box, width, height, margin):
    x1, y1, x2, y2 = box
    bw = max(1, x2 - x1)
    bh = max(1, y2 - y1)
    pad_x = int(round(bw * margin))
    pad_y = int(round(bh * margin))
    return (
        max(0, x1 - pad_x),
        max(0, y1 - pad_y),
        min(width, x2 + pad_x),
        min(height, y2 + pad_y),
    )


def mask_from_box(image_size, box, category):
    width, height = image_size
    mask = Image.new("L", (width, height), 0)
    draw = ImageDraw.Draw(mask)
    x1, y1, x2, y2 = box
    if normalize_class_name(category) in {"ring", "earrings", "watch", "bracelet", "necklace"}:
        draw.ellipse((x1, y1, x2, y2), fill=255)
    else:
        draw.rounded_rectangle((x1, y1, x2, y2), radius=max(2, min(x2 - x1, y2 - y1) // 12), fill=255)
    return mask.filter(ImageFilter.GaussianBlur(radius=max(1, min(width, height) // 400)))


def mask_pixels(mask):
    return int((np.asarray(mask.convert("L")) > 8).sum())


def erase_masked_region(image, mask):
    image = image.convert("RGB")
    alpha = mask.convert("L").filter(ImageFilter.GaussianBlur(radius=5))
    blurred = image.filter(ImageFilter.GaussianBlur(radius=18))
    white = Image.new("RGB", image.size, (255, 255, 255))
    fill = Image.blend(blurred, white, 0.58)
    return Image.composite(fill, image, alpha)


def crop_object(image, mask, box):
    crop = image.crop(box).convert("RGB")
    crop_mask = mask.crop(box).convert("L")
    white = Image.new("RGB", crop.size, (255, 255, 255))
    return Image.composite(crop, white, crop_mask)


def best_object(item):
    objects = item.get("objects", [])
    if not objects:
        return None
    return max(objects, key=lambda obj: float(obj.get("confidence", 0.0) or 0.0))


def build_item(item, asset_dir, raw_dir, args):
    category = normalize_class_name(item.get("category") or item.get("query_category") or "")
    obj = best_object(item)
    if not obj or obj.get("category") == "none":
        return None, "missing_object"
    if obj.get("category"):
        category = normalize_class_name(obj["category"])
    bbox = obj.get("bbox_xyxy_norm")
    if not bbox or len(bbox) != 4:
        return None, "missing_bbox"

    try:
        image_path = download_image(item, raw_dir)
    except Exception as exc:
        return None, f"download_failed:{exc}"
    if not image_path or not image_path.is_file():
        return None, "missing_image"

    try:
        image = Image.open(image_path).convert("RGB")
    except Exception as exc:
        return None, f"open_failed:{exc}"
    box = bbox_to_pixels(bbox, image.width, image.height)
    if box[2] <= box[0] or box[3] <= box[1]:
        return None, "empty_box"
    mask = mask_from_box(image.size, box, category)
    if mask_pixels(mask) < args.min_mask_pixels:
        return None, "small_mask"

    item_dir = Path(asset_dir) / category.replace(" ", "_") / slug(item["id"])
    item_dir.mkdir(parents=True, exist_ok=True)
    target_path = item_dir / "target.jpg"
    mask_path = item_dir / "object_mask.jpg"
    person_path = item_dir / "person_erased.jpg"
    object_path = item_dir / "object_crop.jpg"
    metadata_path = item_dir / "metadata.json"

    crop_box = expand_box(box, image.width, image.height, args.crop_margin)
    image.save(target_path, quality=95)
    mask.save(mask_path, quality=95)
    erase_masked_region(image, mask).save(person_path, quality=95)
    crop_object(image, mask, crop_box).save(object_path, quality=95)

    built = {
        "id": item["id"],
        "source": "wikimedia_commons_llm_pseudo_pair",
        "pairing": "llm_bbox_self_reconstruction",
        "category": category,
        "garment_class": category,
        "class_name": f"commons_{category.replace(' ', '_')}",
        "person_path": str(person_path),
        "object_path": str(object_path),
        "target_path": str(target_path),
        "original_person_path": str(image_path),
        "person_mask_path": str(mask_path),
        "license": item.get("license"),
        "license_url": item.get("license_url"),
        "title": item.get("title"),
        "url": item.get("url"),
        "thumb_url": item.get("thumb_url"),
        "llm_object": obj,
        "gt": {
            "caption": item.get("reason") or obj.get("description") or f"person wearing {category}",
            "caption_cate": f"person wearing {category}. {obj.get('description', '')}".strip(),
        },
        "object": {
            "caption": obj.get("description") or category,
        },
        "person": {
            "caption": item.get("title") or "Commons person image",
        },
    }
    built["hardness_score"] = hardness_score(built)
    write_json(metadata_path, built)
    return built, None


def main():
    args = parse_args()
    payload = load_json(args.input)
    rows = payload.get("items", payload)
    if args.max_items is not None:
        rows = rows[: args.max_items]

    asset_dir = Path(args.asset_dir)
    raw_dir = asset_dir / "_raw"
    items = []
    skipped = []
    for row in rows:
        built, reason = build_item(row, asset_dir, raw_dir, args)
        if built is None:
            skipped.append({"id": row.get("id"), "reason": reason})
            print(f"skip {row.get('id')}: {reason}")
            continue
        items.append(built)
        print(f"built {built['id']} -> {built['category']}")

    out = {
        "source": "wikimedia_commons_llm_pseudo_pair",
        "pairing": "llm_bbox_self_reconstruction",
        "input": args.input,
        "summary": summarize_manifest(items),
        "skipped": skipped,
        "items": items,
    }
    write_json(Path(args.output), out)
    print(f"Wrote {len(items)} Commons pseudo-paired items -> {args.output}")
    print(out["summary"])
    if skipped:
        print(f"Skipped {len(skipped)} items")


if __name__ == "__main__":
    raise SystemExit(main())
