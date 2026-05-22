#!/usr/bin/env python
import argparse
import sys
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from omnitry.enhance.data import (
    demo_items,
    image_exists,
    load_bench_items,
    select_hard_cases,
    summarize_manifest,
    write_json,
)


def parse_args():
    parser = argparse.ArgumentParser(description="Build a hard-case manifest for OmniTry++ training/evaluation.")
    parser.add_argument("--bench-root", default="data/OmniTry_Bench")
    parser.add_argument("--index", default=None)
    parser.add_argument("--output", default="data/hard_cases/omnitry_hard_cases.json")
    parser.add_argument("--top-k", type=int, default=300)
    parser.add_argument("--per-class", type=int, default=0)
    parser.add_argument("--max-items", type=int, default=None)
    parser.add_argument("--demo-fallback", action="store_true", help="Use demo examples if OmniTry-Bench is missing.")
    parser.add_argument("--require-local-images", action="store_true", help="Keep only items whose person/object images exist locally.")
    return parser.parse_args()


def main():
    args = parse_args()
    bench_root = Path(args.bench_root)

    try:
        items = load_bench_items(bench_root, Path(args.index) if args.index else None, max_items=args.max_items)
        source = "omnitry_bench"
    except FileNotFoundError:
        if not args.demo_fallback:
            raise
        items = demo_items(Path("."))
        source = "demo_example"

    selected = select_hard_cases(items, top_k=args.top_k, per_class=args.per_class)
    if args.require_local_images:
        selected = [item for item in selected if image_exists(item)]
        if not selected and args.demo_fallback:
            selected = select_hard_cases(demo_items(Path(".")), top_k=args.top_k, per_class=args.per_class)
            source = "demo_example"
    payload = {
        "source": source,
        "summary": summarize_manifest(selected),
        "items": selected,
    }
    write_json(Path(args.output), payload)
    print(f"Wrote {len(selected)} hard cases -> {args.output}")
    print(payload["summary"])


if __name__ == "__main__":
    raise SystemExit(main())
