#!/usr/bin/env python
import argparse
import json
import shutil
from pathlib import Path


def parse_args():
    parser = argparse.ArgumentParser(description="Build a best-of-N benchmark summary from existing benchmark outputs.")
    parser.add_argument("--summary", action="append", required=True, help="Input benchmark summary JSON. Repeatable.")
    parser.add_argument("--label", action="append", required=True, help="Label for each input summary. Repeatable.")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--summary-output", required=True)
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
        return {}
    keys = ["total", "object", "person", "artifact"]
    summary = {f"{key}_mean": round(sum(row[key] for row in rows) / len(rows), 6) for key in keys}
    classes = {}
    for row in rows:
        classes.setdefault(row.get("category", "unknown"), []).append(row)
    summary["items"] = len(rows)
    summary["classes"] = {
        category: {
            "count": len(class_rows),
            **{
                f"{key}_mean": round(sum(row[key] for row in class_rows) / len(class_rows), 6)
                for key in keys
            },
        }
        for category, class_rows in sorted(classes.items())
    }
    return summary


def copy_artifact(src, output_dir, suffix):
    src_path = Path(src)
    if not src_path.is_file():
        return str(src)
    dst = output_dir / f"{src_path.stem}{suffix}{src_path.suffix}"
    shutil.copy2(src_path, dst)
    return str(dst)


def main():
    args = parse_args()
    if len(args.summary) != len(args.label):
        raise ValueError("--summary and --label must have the same count.")

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    labeled_items = []
    for label, summary_path in zip(args.label, args.summary):
        payload = load_json(summary_path)
        if payload.get("complete") is False:
            raise ValueError(f"Input summary is incomplete: {summary_path}")
        for item in payload.get("items", []):
            row = dict(item)
            row["_source_label"] = label
            row["_source_summary"] = summary_path
            labeled_items.append(row)

    by_id = {}
    for item in labeled_items:
        by_id.setdefault(item["id"], []).append(item)

    rows = []
    source_counts = {}
    for item_id in sorted(by_id):
        candidates = by_id[item_id]
        if len(candidates) < len(args.summary):
            continue
        best = max(candidates, key=lambda row: row["total"])
        source_label = best["_source_label"]
        source_counts[source_label] = source_counts.get(source_label, 0) + 1

        row = {
            key: value
            for key, value in best.items()
            if key not in {"_source_label", "_source_summary"}
        }
        suffix = f"__{source_label}"
        row["source"] = source_label
        row["source_summary"] = best["_source_summary"]
        row["image"] = copy_artifact(row["image"], output_dir, suffix)
        row["diagnostics"] = copy_artifact(row["diagnostics"], output_dir, suffix)
        rows.append(row)

    payload = {
        "mode": "hybrid_best_total",
        "inputs": [{"label": label, "summary": path} for label, path in zip(args.label, args.summary)],
        "output_dir": str(output_dir),
        "source_counts": source_counts,
        "summary": summarize(rows),
        "items": rows,
        "complete": True,
    }
    write_json(args.summary_output, payload)
    print(json.dumps(payload["summary"], indent=2))
    print(f"source_counts={source_counts}")
    print(f"Wrote hybrid summary -> {args.summary_output}")


if __name__ == "__main__":
    raise SystemExit(main())
