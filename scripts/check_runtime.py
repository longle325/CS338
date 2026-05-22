#!/usr/bin/env python
import os
from pathlib import Path

import torch
from omegaconf import OmegaConf


MIN_RECOMMENDED_VRAM_GB = float(os.environ.get("OMNITRY_MIN_VRAM_GB", "28"))


def _format_gb(num_bytes: int) -> float:
    return num_bytes / (1024 ** 3)


def main():
    config = OmegaConf.load("configs/omnitry_v1_unified.yaml")
    model_root = Path(config.model_root)
    lora_path = Path(config.lora_path)

    print("OmniTry runtime check")
    print(f"- model_root: {model_root}")
    print(f"- lora_path: {lora_path}")

    missing = []
    if not (model_root / "transformer").exists():
        missing.append(model_root / "transformer")
    if not lora_path.is_file():
        missing.append(lora_path)

    if torch.cuda.is_available():
        count = torch.cuda.device_count()
        print(f"- CUDA: yes ({count} GPU(s))")
        for index in range(count):
            props = torch.cuda.get_device_properties(index)
            total_gb = _format_gb(props.total_memory)
            marker = "ok" if total_gb >= MIN_RECOMMENDED_VRAM_GB else "low-vram"
            print(f"  [{index}] {props.name}: {total_gb:.1f} GB ({marker})")
    else:
        print("- CUDA: no")

    if missing:
        print("- checkpoints: missing")
        for path in missing:
            print(f"  - {path}")
        return 2

    print("- checkpoints: ok")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
