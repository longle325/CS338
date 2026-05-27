#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch
import torchvision
from PIL import Image, ImageFile
from segment_anything import SamPredictor, sam_model_registry
from tqdm import tqdm
from transformers import GroundingDinoForObjectDetection, GroundingDinoProcessor

ImageFile.LOAD_TRUNCATED_IMAGES = True

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate OmniTry-Bench official masks with GroundingDINO + SAM.")
    parser.add_argument("--benchmark-file", default="data/OmniTry_Bench/omni_vtryon_bench_small_v1.json")
    parser.add_argument("--bench-root", default="data/OmniTry_Bench")
    parser.add_argument("--result-dir", required=True)
    parser.add_argument("--summary-output", default=None)
    parser.add_argument("--groundingdino-root", default="omnitry_bench/Grounded-Segment-Anything/GroundingDINO")
    parser.add_argument("--groundingdino-config", default=None)
    parser.add_argument("--groundingdino-checkpoint", default="checkpoints/groundingdino_swint_ogc.pth")
    parser.add_argument("--groundingdino-model", default="IDEA-Research/grounding-dino-tiny")
    parser.add_argument("--sam-checkpoint", default="checkpoints/sam_vit_h_4b8939.pth")
    parser.add_argument("--sam-encoder", default="vit_h")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--box-threshold", type=float, default=0.25)
    parser.add_argument("--fallback-box-threshold", type=float, default=0.10)
    parser.add_argument("--text-threshold", type=float, default=0.25)
    parser.add_argument("--nms-threshold", type=float, default=0.80)
    parser.add_argument("--max-items", type=int, default=None)
    parser.add_argument("--skip-existing", action="store_true")
    return parser.parse_args()


def load_items(path: Path, max_items: int | None) -> list[dict]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    items = payload["items"] if isinstance(payload, dict) and "items" in payload else payload
    return items[:max_items] if max_items else items


def resolve_bench_path(raw_path: str, bench_root: Path) -> Path:
    path = Path(raw_path)
    if path.is_file():
        return path
    parts = path.parts
    if parts and parts[0] == bench_root.name:
        candidate = bench_root.parent / path
    else:
        candidate = bench_root / path
    return candidate


def result_path_for(item: dict, result_dir: Path) -> Path:
    path = result_dir / f"{item['id']}.jpg"
    if path.is_file():
        return path
    return result_dir / f"{item['class_name']}_{item['id']}.jpg"


def mask_path_for(image_path: Path) -> Path:
    return image_path.with_name(f"{image_path.stem}_mask.jpg")


class GroundingDinoHF:
    def __init__(self, model_name: str, device: torch.device) -> None:
        self.device = device
        self.processor = GroundingDinoProcessor.from_pretrained(model_name)
        self.model = GroundingDinoForObjectDetection.from_pretrained(model_name).to(device).eval()

    @torch.no_grad()
    def predict_boxes(
        self,
        image_pil: Image.Image,
        prompt: str,
        box_threshold: float,
        text_threshold: float,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        text = prompt.strip()
        if not text.endswith("."):
            text = f"{text}."
        inputs = self.processor(images=image_pil, text=text, return_tensors="pt").to(self.device)
        outputs = self.model(**inputs)
        target_sizes = torch.tensor([(image_pil.height, image_pil.width)], device=self.device)
        result = self.processor.post_process_grounded_object_detection(
            outputs,
            inputs["input_ids"],
            box_threshold=box_threshold,
            text_threshold=text_threshold,
            target_sizes=target_sizes,
        )[0]
        boxes = result["boxes"].detach().cpu().float()
        scores = result.get("scores", torch.ones(len(boxes))).detach().cpu().float()
        return boxes, scores


def detections_to_mask(
    *,
    model,
    sam_predictor: SamPredictor,
    image_path: Path,
    prompt: str,
    box_threshold: float,
    text_threshold: float,
    nms_threshold: float,
) -> torch.Tensor:
    image_pil = Image.open(image_path).convert("RGB")
    image_rgb = np.array(image_pil)

    boxes, scores = model.predict_boxes(
        image_pil=image_pil,
        prompt=prompt,
        box_threshold=box_threshold,
        text_threshold=text_threshold,
    )
    if len(boxes) == 0:
        return torch.zeros((1, image_pil.height, image_pil.width), dtype=torch.float32)

    nms_idx = torchvision.ops.nms(boxes, scores, nms_threshold).numpy().tolist()
    boxes = boxes[nms_idx]

    topk = 2 if "earring" in prompt else 1
    boxes = boxes[:topk].numpy()
    if len(boxes) == 0:
        return torch.zeros((1, image_pil.height, image_pil.width), dtype=torch.float32)

    sam_predictor.set_image(image_rgb)
    masks = []
    for box in boxes:
        candidate_masks, scores, _logits = sam_predictor.predict(box=box, multimask_output=True)
        masks.append(candidate_masks[int(np.argmax(scores))])
    if not masks:
        return torch.zeros((1, image_pil.height, image_pil.width), dtype=torch.float32)
    return torch.tensor(np.any(np.stack(masks, axis=0), axis=0, keepdims=True), dtype=torch.float32)


def save_mask(mask: torch.Tensor, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    torchvision.utils.save_image(mask, path)


def main() -> int:
    args = parse_args()
    benchmark_file = ROOT / args.benchmark_file
    bench_root = ROOT / args.bench_root
    result_dir = ROOT / args.result_dir
    sam_checkpoint = ROOT / args.sam_checkpoint
    summary_output = Path(args.summary_output) if args.summary_output else result_dir / "official_mask_summary.json"

    device = torch.device(args.device if torch.cuda.is_available() and args.device.startswith("cuda") else "cpu")
    grounding_model = GroundingDinoHF(args.groundingdino_model, device)
    sam = sam_model_registry[args.sam_encoder](checkpoint=str(sam_checkpoint))
    sam.to(device=device)
    sam_predictor = SamPredictor(sam)

    items = load_items(benchmark_file, args.max_items)
    processed = 0
    missing_results = []
    none_masks = 0
    fallback_masks = 0
    errors = []

    for item in tqdm(items, desc="official masks"):
        prompt = item["garment_class"]
        object_path = resolve_bench_path(item["object"]["img_path"], bench_root)
        tryon_path = result_path_for(item, result_dir)
        if not tryon_path.is_file():
            missing_results.append(item["id"])
            continue

        for image_path, label in [(object_path, "object"), (tryon_path, "tryon")]:
            out_path = mask_path_for(image_path)
            if args.skip_existing and out_path.is_file():
                continue
            try:
                mask = detections_to_mask(
                    model=grounding_model,
                    sam_predictor=sam_predictor,
                    image_path=image_path,
                    prompt=prompt,
                    box_threshold=args.box_threshold,
                    text_threshold=args.text_threshold,
                    nms_threshold=args.nms_threshold,
                )
                if label == "tryon" and mask.eq(0).all():
                    fallback = detections_to_mask(
                        model=grounding_model,
                        sam_predictor=sam_predictor,
                        image_path=image_path,
                        prompt=prompt,
                        box_threshold=args.fallback_box_threshold,
                        text_threshold=args.text_threshold,
                        nms_threshold=args.nms_threshold,
                    )
                    if not fallback.eq(0).all():
                        mask = fallback
                        fallback_masks += 1
                if label == "tryon" and mask.eq(0).all():
                    none_masks += 1
                save_mask(mask, out_path)
            except Exception as exc:  # noqa: BLE001
                errors.append({"id": item["id"], "image": str(image_path), "label": label, "error": str(exc)})
        processed += 1

    summary = {
        "benchmark_file": str(benchmark_file),
        "result_dir": str(result_dir),
        "items": len(items),
        "processed": processed,
        "missing_results": missing_results,
        "tryon_none_masks": none_masks,
        "tryon_fallback_masks": fallback_masks,
        "errors": errors,
        "complete": processed == len(items) and not missing_results and not errors,
    }
    summary_output.parent.mkdir(parents=True, exist_ok=True)
    summary_output.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))
    return 0 if summary["complete"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
