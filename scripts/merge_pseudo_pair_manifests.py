#!/usr/bin/env python
import argparse
import json
import sys
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from omnitry.enhance.data import summarize_manifest, write_json


def parse_args():
    parser = argparse.ArgumentParser(description="Merge pseudo-paired manifests, optionally repeating later manifests.")
    parser.add_argument("--manifest", action="append", required=True)
    parser.add_argument("--repeat", action="append", type=int, default=None)
    parser.add_argument("--output", required=True)
    return parser.parse_args()


def load_items(path):
    payload = json.load(Path(path).open("r", encoding="utf-8"))
    return payload.get("items", payload)


def main():
    args = parse_args()
    repeats = args.repeat or [1] * len(args.manifest)
    if len(repeats) != len(args.manifest):
        raise ValueError("--repeat count must match --manifest count")

    merged = []
    seen = set()
    input_summaries = []
    for manifest, repeat in zip(args.manifest, repeats):
        rows = load_items(manifest)
        input_summaries.append({"manifest": manifest, "items": len(rows), "repeat": repeat})
        for rep in range(max(1, repeat)):
            for row in rows:
                copied = dict(row)
                if repeat > 1:
                    copied["id"] = f"{copied.get('id', 'item')}__r{rep:02d}"
                    copied["repeat_source_id"] = row.get("id")
                item_id = copied.get("id")
                if item_id in seen:
                    suffix = len(seen)
                    copied["id"] = f"{item_id}__dup{suffix}"
                seen.add(copied["id"])
                merged.append(copied)

    payload = {
        "source": "merged_pseudo_pair_manifest",
        "pairing": "mixed_self_reconstruction",
        "inputs": input_summaries,
        "summary": summarize_manifest(merged),
        "items": merged,
    }
    write_json(Path(args.output), payload)
    print(f"Wrote {len(merged)} merged items -> {args.output}")
    print(payload["summary"])


if __name__ == "__main__":
    raise SystemExit(main())
