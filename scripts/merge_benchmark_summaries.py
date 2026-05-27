#!/usr/bin/env python
import argparse
import json
from pathlib import Path


def parse_args():
    parser = argparse.ArgumentParser(description="Merge sharded run_tryon_benchmark summaries.")
    parser.add_argument("--summary", action="append", required=True)
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--summary-output", required=True)
    parser.add_argument("--expected-count", type=int, default=None)
    return parser.parse_args()


def load_json(path):
    with Path(path).open("r", encoding="utf-8") as handle:
        return json.load(handle)


def write_json(path, payload):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2)
        handle.write("\n")


def summarize(rows):
    if not rows:
        return {"items": 0}
    keys = ["total", "object", "person", "artifact"]
    summary = {
        "items": len(rows),
        **{f"{key}_mean": round(sum(row[key] for row in rows) / len(rows), 6) for key in keys},
    }
    by_class = {}
    for row in rows:
        by_class.setdefault(row.get("category", "unknown"), []).append(row)
    summary["classes"] = {
        category: {
            "count": len(class_rows),
            **{f"{key}_mean": round(sum(row[key] for row in class_rows) / len(class_rows), 6) for key in keys},
        }
        for category, class_rows in sorted(by_class.items())
    }
    return summary


def manifest_order(path):
    payload = load_json(path)
    items = payload.get("items", payload) if isinstance(payload, dict) else payload
    return {item["id"]: index for index, item in enumerate(items)}


def main():
    args = parse_args()
    order = manifest_order(args.manifest)
    by_id = {}
    inputs = []
    complete = True
    mode = None
    lora_path = None

    for summary_path in args.summary:
        payload = load_json(summary_path)
        inputs.append(summary_path)
        complete = complete and bool(payload.get("complete"))
        mode = mode or payload.get("mode")
        lora_path = lora_path or payload.get("lora_path")
        for item in payload.get("items", []):
            by_id[item["id"]] = item

    rows = sorted(by_id.values(), key=lambda row: order.get(row["id"], 10**9))
    if args.expected_count is not None:
        complete = complete and len(rows) >= args.expected_count

    output = {
        "manifest": args.manifest,
        "mode": mode,
        "lora_path": lora_path,
        "output_dir": args.output_dir,
        "inputs": inputs,
        "summary": summarize(rows),
        "items": rows,
        "complete": complete,
    }
    write_json(args.summary_output, output)
    print(json.dumps(output["summary"], indent=2))
    print(f"complete={complete}")
    print(f"Wrote merged summary -> {args.summary_output}")


if __name__ == "__main__":
    raise SystemExit(main())
