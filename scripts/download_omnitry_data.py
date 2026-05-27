#!/usr/bin/env python
import argparse
import os
from pathlib import Path

from huggingface_hub import get_token, snapshot_download


def parse_args():
    parser = argparse.ArgumentParser(description="Download OmniTry-Bench metadata or full dataset.")
    parser.add_argument("--repo-id", default="Kunbyte/OmniTry-Bench")
    parser.add_argument("--output-dir", default="data/OmniTry_Bench")
    parser.add_argument("--full", action="store_true", help="Download images as well as JSON metadata.")
    parser.add_argument("--max-workers", type=int, default=4, help="Parallel Hugging Face download workers.")
    return parser.parse_args()


def main():
    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    allow_patterns = None if args.full else ["*.json", "**/*.json", "README*", "*.md"]
    print(f"Downloading dataset repo {args.repo_id} -> {output_dir}")
    if allow_patterns:
        print("Mode: metadata-only. Use --full for images.")

    token = os.environ.get("HF_TOKEN") or os.environ.get("HUGGING_FACE_HUB_TOKEN") or get_token()
    if token:
        print("Using Hugging Face authentication from environment/cache.")

    snapshot_download(
        repo_id=args.repo_id,
        repo_type="dataset",
        local_dir=str(output_dir),
        allow_patterns=allow_patterns,
        resume_download=True,
        token=token,
        max_workers=args.max_workers,
    )
    print("OmniTry-Bench download step complete.")


if __name__ == "__main__":
    raise SystemExit(main())
