#!/usr/bin/env python
import argparse
import json
import sys
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

import numpy as np
from PIL import Image, ImageFile

ImageFile.LOAD_TRUNCATED_IMAGES = True

from gradio_demo import args as demo_args
from gradio_demo import generate, validate_checkpoint_paths
from omnitry.enhance import confidence_label, score_candidate
from omnitry.enhance.data import load_json, write_json


def parse_args():
    parser = argparse.ArgumentParser(description="Run Baseline/Enhanced OmniTry generation benchmark.")
    parser.add_argument("--manifest", default="data/hard_cases/omnitry_full_local_hard_cases.json")
    parser.add_argument("--output-dir", default="outputs/tryon_benchmark/enhanced")
    parser.add_argument("--summary-output", default="outputs/tryon_benchmark/enhanced_summary.json")
    parser.add_argument("--mode", choices=["Enhanced", "Baseline"], default="Enhanced")
    parser.add_argument("--max-items", type=int, default=32)
    parser.add_argument("--start-index", type=int, default=0)
    parser.add_argument("--end-index", type=int, default=None)
    parser.add_argument("--steps", type=int, default=20)
    parser.add_argument("--guidance-scale", type=float, default=30.0)
    parser.add_argument("--candidate-count", type=int, default=1)
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument("--lora-path", default=None, help="Override the demo LoRA checkpoint path.")
    parser.add_argument("--skip-existing", action="store_true")
    return parser.parse_args()


def load_items(manifest, max_items=None, start_index=0, end_index=None):
    payload = load_json(Path(manifest))
    items = payload["items"] if isinstance(payload, dict) and "items" in payload else payload
    if max_items is not None:
        items = items[:max_items]
    start_index = max(0, int(start_index or 0))
    if end_index is None:
        end_index = len(items)
    end_index = min(len(items), int(end_index))
    return list(enumerate(items[start_index:end_index], start=start_index))


def summarize(rows):
    if not rows:
        return {}
    keys = ["total", "object", "person", "artifact"]
    means = {f"{key}_mean": round(float(np.mean([row[key] for row in rows])), 6) for key in keys}
    by_class = {}
    for row in rows:
        category = row.get("category", "unknown")
        by_class.setdefault(category, []).append(row)
    means["classes"] = {
        category: {
            "count": len(class_rows),
            "total_mean": round(float(np.mean([row["total"] for row in class_rows])), 6),
            "object_mean": round(float(np.mean([row["object"] for row in class_rows])), 6),
            "person_mean": round(float(np.mean([row["person"] for row in class_rows])), 6),
            "artifact_mean": round(float(np.mean([row["artifact"] for row in class_rows])), 6),
        }
        for category, class_rows in sorted(by_class.items())
    }
    return {"items": len(rows), **means}


def main():
    args = parse_args()
    if args.lora_path:
        demo_args.lora_path = args.lora_path
    validate_checkpoint_paths()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    rows = []
    for index, item in load_items(args.manifest, args.max_items, args.start_index, args.end_index):
        item_id = item.get("id", f"item_{index:06d}")
        category = item.get("category") or item.get("garment_class")
        out_path = output_dir / f"{item_id}.jpg"
        diag_path = output_dir / f"{item_id}.md"

        person = Image.open(item["person_path"]).convert("RGB")
        obj = Image.open(item["object_path"]).convert("RGB")

        if args.skip_existing and out_path.is_file():
            generated = Image.open(out_path).convert("RGB")
            diagnostics = diag_path.read_text(encoding="utf-8") if diag_path.is_file() else ""
        else:
            generated, _gallery, diagnostics = generate(
                person_image=person,
                object_image=obj,
                object_class=category,
                optional_prompt="",
                steps=args.steps,
                guidance_scale=args.guidance_scale,
                seed=args.seed + index,
                candidate_count=args.candidate_count,
                enhance_mode=args.mode,
                progress=None,
            )
            generated.save(out_path, quality=95)
            diag_path.write_text(diagnostics, encoding="utf-8")

        total, object_score, person_score, artifact_score = score_candidate(generated, person, obj, category)
        rows.append(
            {
                "id": item_id,
                "category": category,
                "image": str(out_path),
                "diagnostics": str(diag_path),
                "total": round(total, 6),
                "object": round(object_score, 6),
                "person": round(person_score, 6),
                "artifact": round(artifact_score, 6),
                "confidence": confidence_label(total),
            }
        )
        print(rows[-1])
        write_json(
            Path(args.summary_output),
            {
                "manifest": args.manifest,
                "mode": args.mode,
                "lora_path": args.lora_path or str(demo_args.lora_path),
                "output_dir": str(output_dir),
                "summary": summarize(rows),
                "items": rows,
                "complete": False,
            },
        )

    payload = {
        "manifest": args.manifest,
        "mode": args.mode,
        "lora_path": args.lora_path or str(demo_args.lora_path),
        "output_dir": str(output_dir),
        "summary": summarize(rows),
        "items": rows,
        "complete": True,
    }
    write_json(Path(args.summary_output), payload)
    print(payload["summary"])
    print(f"Wrote benchmark summary -> {args.summary_output}")


if __name__ == "__main__":
    raise SystemExit(main())
