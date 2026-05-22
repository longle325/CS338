#!/usr/bin/env python
import argparse
from pathlib import Path

from huggingface_hub import snapshot_download


def parse_args():
    parser = argparse.ArgumentParser(description="Download OmniTry-Bench metadata or full dataset.")
    parser.add_argument("--repo-id", default="Kunbyte/OmniTry-Bench")
    parser.add_argument("--output-dir", default="data/OmniTry_Bench")
    parser.add_argument("--full", action="store_true", help="Download images as well as JSON metadata.")
    return parser.parse_args()


def main():
    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    allow_patterns = None if args.full else ["*.json", "**/*.json", "README*", "*.md"]
    print(f"Downloading dataset repo {args.repo_id} -> {output_dir}")
    if allow_patterns:
        print("Mode: metadata-only. Use --full for images.")

    snapshot_download(
        repo_id=args.repo_id,
        repo_type="dataset",
        local_dir=str(output_dir),
        allow_patterns=allow_patterns,
        resume_download=True,
    )
    print("OmniTry-Bench download step complete.")


if __name__ == "__main__":
    raise SystemExit(main())
