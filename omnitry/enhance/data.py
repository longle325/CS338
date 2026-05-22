import json
import math
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
from PIL import Image

from .taxonomy import AFFORDANCE_BOXES, HARD_CASE_CLASSES, normalize_class_name


BENCH_INDEX_CANDIDATES = (
    "omni_vtryon_bench_small_v1.json",
    "omni_vtryon_bench_v1.json",
    "omni_vtryon_benchmark_small_v1.json",
    "omni_vtryon_benchmark_v1.json",
)


def find_bench_index(root: Path, prefer_small: bool = True) -> Path:
    root = Path(root)
    candidates = BENCH_INDEX_CANDIDATES if prefer_small else tuple(reversed(BENCH_INDEX_CANDIDATES))
    for name in candidates:
        path = root / name
        if path.is_file():
            return path
    matches = sorted(root.glob("*.json"))
    if matches:
        return matches[0]
    raise FileNotFoundError(f"No OmniTry-Bench JSON index found under {root}")


def load_json(path: Path):
    with Path(path).open("r", encoding="utf-8") as handle:
        return json.load(handle)


def write_json(path: Path, data):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(data, handle, indent=2, ensure_ascii=False)
        handle.write("\n")


def load_bench_items(root: Path, index_path: Optional[Path] = None, max_items: Optional[int] = None) -> List[Dict]:
    root = Path(root)
    index_path = Path(index_path) if index_path else find_bench_index(root)
    items = load_json(index_path)
    if max_items is not None:
        items = items[:max_items]

    enriched = []
    for item in items:
        copied = dict(item)
        copied["source"] = "omnitry_bench"
        copied["index_path"] = str(index_path)
        copied["person_path"] = _resolve_bench_path(root, item.get("person", {}).get("img_path", ""))
        copied["object_path"] = _resolve_bench_path(root, item.get("object", {}).get("img_path", ""))
        copied["category"] = normalize_class_name(item.get("garment_class") or item.get("class_name", "").split("_")[0])
        copied["hardness_score"] = hardness_score(copied)
        enriched.append(copied)
    return enriched


def _resolve_bench_path(root: Path, raw_path: str) -> str:
    raw = Path(raw_path)
    if raw.is_absolute():
        return str(raw)
    direct = root / raw
    if direct.exists():
        return str(direct)
    if raw.parts and raw.parts[0] == root.name:
        without_root = root.joinpath(*raw.parts[1:])
        if without_root.exists():
            return str(without_root)
    return str(direct)


def hardness_score(item: Dict) -> float:
    category = normalize_class_name(item.get("category") or item.get("garment_class", ""))
    class_name = item.get("class_name", "")
    person_caption = item.get("person", {}).get("caption", "")
    object_caption = item.get("object", {}).get("caption", "")
    text = f"{class_name} {person_caption} {object_caption}".lower()

    score = 0.0
    if category in HARD_CASE_CLASSES:
        score += 2.0
    if category in {"ring", "earrings", "bracelet", "watch", "necklace"}:
        score += 1.5
    if category in {"bag", "shoe", "hat", "glasses", "sunglasses"}:
        score += 0.8
    for keyword, weight in {
        "wild": 0.8,
        "natural": 0.8,
        "side": 0.6,
        "profile": 0.7,
        "back": 0.6,
        "hand": 0.7,
        "holding": 0.9,
        "occlusion": 1.0,
        "hair": 0.4,
        "small": 0.6,
        "logo": 0.7,
        "metal": 0.5,
        "reflect": 0.5,
        "transparent": 0.5,
    }.items():
        if keyword in text:
            score += weight
    return round(score, 3)


def select_hard_cases(items: Sequence[Dict], top_k: int, per_class: int = 0) -> List[Dict]:
    ranked = sorted(items, key=lambda item: item.get("hardness_score", 0.0), reverse=True)
    if per_class <= 0:
        return ranked[:top_k]

    selected = []
    counts: Dict[str, int] = {}
    for item in ranked:
        category = item.get("category", "unknown")
        if counts.get(category, 0) >= per_class:
            continue
        selected.append(item)
        counts[category] = counts.get(category, 0) + 1
        if len(selected) >= top_k:
            break
    return selected


def demo_items(root: Path = Path(".")) -> List[Dict]:
    examples = [
        ("ring", "demo_example/person_ring.jpg", "demo_example/object_ring.jpg"),
        ("earrings", "demo_example/person_earrings.jpg", "demo_example/object_earrings.jpg"),
        ("bracelet", "demo_example/person_bracelet.jpg", "demo_example/object_bracelet.jpg"),
        ("necklace", "demo_example/person_necklace.jpg", "demo_example/object_necklace.jpg"),
        ("glasses", "demo_example/person_glasses.jpg", "demo_example/object_glasses.jpg"),
        ("sunglasses", "demo_example/person_sunglasses.jpg", "demo_example/object_sunglasses.jpg"),
        ("bag", "demo_example/person_bag.jpg", "demo_example/object_bag.jpg"),
        ("shoe", "demo_example/person_shoes.jpg", "demo_example/object_shoes.jpg"),
        ("hat", "demo_example/person_hat.jpg", "demo_example/object_hat.jpg"),
        ("tie", "demo_example/person_tie.jpg", "demo_example/object_tie.jpg"),
        ("bow tie", "demo_example/person_bowtie.jpg", "demo_example/object_bowtie.jpg"),
    ]
    items = []
    for index, (category, person_path, object_path) in enumerate(examples):
        item = {
            "id": f"demo_{category.replace(' ', '_')}_{index:03d}",
            "source": "demo_example",
            "category": category,
            "garment_class": category,
            "class_name": category.replace(" ", "_"),
            "person_path": str(root / person_path),
            "object_path": str(root / object_path),
            "person": {"img_path": person_path, "caption": "demo person image"},
            "object": {"img_path": object_path, "caption": f"demo {category} object"},
        }
        item["hardness_score"] = hardness_score(item)
        items.append(item)
    return items


def normalized_box(category: str, width: int, height: int) -> Tuple[int, int, int, int]:
    x1, y1, x2, y2 = AFFORDANCE_BOXES.get(normalize_class_name(category), (0.0, 0.0, 1.0, 1.0))
    return (
        max(0, min(width - 1, int(x1 * width))),
        max(0, min(height - 1, int(y1 * height))),
        max(1, min(width, int(x2 * width))),
        max(1, min(height, int(y2 * height))),
    )


def weak_heatmap(category: str, size: int = 64) -> np.ndarray:
    x1, y1, x2, y2 = normalized_box(category, size, size)
    heatmap = np.zeros((size, size), dtype=np.float32)
    cx = (x1 + x2) / 2.0
    cy = (y1 + y2) / 2.0
    sigma_x = max(2.0, (x2 - x1) / 3.0)
    sigma_y = max(2.0, (y2 - y1) / 3.0)
    yy, xx = np.mgrid[0:size, 0:size]
    heatmap = np.exp(-(((xx - cx) ** 2) / (2 * sigma_x**2) + ((yy - cy) ** 2) / (2 * sigma_y**2)))
    heatmap[y1:y2, x1:x2] = np.maximum(heatmap[y1:y2, x1:x2], 0.75)
    return heatmap.astype(np.float32)


def image_exists(item: Dict) -> bool:
    return Path(item.get("person_path", "")).is_file() and Path(item.get("object_path", "")).is_file()


def load_person_image(item: Dict, image_size: int):
    image = Image.open(item["person_path"]).convert("RGB")
    image.thumbnail((image_size, image_size))
    canvas = Image.new("RGB", (image_size, image_size), (255, 255, 255))
    left = (image_size - image.width) // 2
    top = (image_size - image.height) // 2
    canvas.paste(image, (left, top))
    return canvas


def dice_score(pred: np.ndarray, target: np.ndarray, threshold: float = 0.5) -> float:
    pred_mask = pred >= threshold
    target_mask = target >= threshold
    inter = np.logical_and(pred_mask, target_mask).sum()
    denom = pred_mask.sum() + target_mask.sum()
    if denom == 0:
        return 1.0
    return float(2 * inter / denom)


def iou_score(pred: np.ndarray, target: np.ndarray, threshold: float = 0.5) -> float:
    pred_mask = pred >= threshold
    target_mask = target >= threshold
    inter = np.logical_and(pred_mask, target_mask).sum()
    union = np.logical_or(pred_mask, target_mask).sum()
    if union == 0:
        return 1.0
    return float(inter / union)


def summarize_manifest(items: Iterable[Dict]) -> Dict:
    items = list(items)
    by_class: Dict[str, int] = {}
    for item in items:
        category = item.get("category", "unknown")
        by_class[category] = by_class.get(category, 0) + 1
    scores = [item.get("hardness_score", 0.0) for item in items]
    return {
        "count": len(items),
        "classes": by_class,
        "hardness_mean": round(float(np.mean(scores)) if scores else math.nan, 3),
        "hardness_max": round(float(np.max(scores)) if scores else math.nan, 3),
    }
