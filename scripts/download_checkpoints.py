#!/usr/bin/env python
import os
import sys
from pathlib import Path

from huggingface_hub import get_token, hf_hub_download, snapshot_download
from huggingface_hub.errors import GatedRepoError, HfHubHTTPError


FLUX_REPO = "black-forest-labs/FLUX.1-Fill-dev"
OMNITRY_REPO = "Kunbyte/OmniTry"
LORA_FILENAME = "omnitry_v1_unified.safetensors"


def _token():
    return os.environ.get("HF_TOKEN") or os.environ.get("HUGGING_FACE_HUB_TOKEN") or get_token()


def _download_flux(model_root: Path, token: str | None):
    print(f"Downloading {FLUX_REPO} -> {model_root}")
    snapshot_download(
        repo_id=FLUX_REPO,
        local_dir=str(model_root),
        token=token,
        resume_download=True,
    )


def _download_lora(lora_path: Path, token: str | None):
    print(f"Downloading {OMNITRY_REPO}/{LORA_FILENAME} -> {lora_path}")
    lora_path.parent.mkdir(parents=True, exist_ok=True)
    downloaded = Path(
        hf_hub_download(
            repo_id=OMNITRY_REPO,
            filename=LORA_FILENAME,
            local_dir=str(lora_path.parent),
            token=token,
            resume_download=True,
        )
    )
    if downloaded != lora_path:
        lora_path.write_bytes(downloaded.read_bytes())


def main():
    model_root = Path(os.environ.get("OMNITRY_MODEL_ROOT", "checkpoints/FLUX.1-Fill-dev"))
    lora_path = Path(os.environ.get("OMNITRY_LORA_PATH", f"checkpoints/{LORA_FILENAME}"))
    token = _token()

    if token is None:
        print(
            "HF_TOKEN is not set. FLUX.1-Fill-dev is a gated Hugging Face model, so the download may fail.\n"
            "Create a Hugging Face token, accept the FLUX.1-Fill-dev license, then run:\n"
            "  export HF_TOKEN=hf_...\n"
            "  bash scripts/setup_omnitry.sh",
            file=sys.stderr,
        )

    try:
        _download_flux(model_root, token)
        _download_lora(lora_path, token)
    except GatedRepoError as exc:
        print(
            f"Access denied for a gated Hugging Face repo: {exc}\n\n"
            "Make sure your Hugging Face account accepted the FLUX.1-Fill-dev license and that "
            "HF_TOKEN is exported or `huggingface-cli login` is configured.",
            file=sys.stderr,
        )
        return 2
    except HfHubHTTPError as exc:
        print(f"Hugging Face download failed: {exc}", file=sys.stderr)
        return 2

    print("Checkpoint download complete.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
