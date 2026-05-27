#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

import lpips
import torch
import torch.nn.functional as F
import torchvision
import torchvision.transforms as T
from huggingface_hub import snapshot_download
from PIL import Image, ImageFile
from torchmetrics.image import StructuralSimilarityIndexMeasure
from tqdm import tqdm
from transformers import CLIPModel, CLIPProcessor, ViTImageProcessor, ViTModel

ImageFile.LOAD_TRUNCATED_IMAGES = True
RESAMPLE_BICUBIC = Image.Resampling.BICUBIC if hasattr(Image, "Resampling") else Image.BICUBIC
RESAMPLE_NEAREST = Image.Resampling.NEAREST if hasattr(Image, "Resampling") else Image.NEAREST

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


PROMPT_POS = {
    "earrings": "on ears",
    "earring": "on ear",
    "necklace": "around neck",
    "bracelet": "around wrist",
    "ring": "on finger",
    "hat": "on head",
    "glasses": "on face",
    "sunglasses": "on face",
    "tie": "around collar",
    "bow tie": "around collar",
    "belt": "around waist",
    "bag": "on shoulder or in hand",
    "dress": "on body",
    "top cloth": "on upper body",
    "top clothes": "on upper body",
    "bottom cloth": "on lower body",
    "bottom clothes": "on lower body",
    "shoes": "on feet",
    "shoe": "on foot",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Compute OmniTry-Bench official metrics.")
    parser.add_argument("--benchmark-file", default="data/OmniTry_Bench/omni_vtryon_bench_small_v1.json")
    parser.add_argument("--bench-root", default="data/OmniTry_Bench")
    parser.add_argument("--result-dir", required=True)
    parser.add_argument("--result-json", default=None)
    parser.add_argument("--detail-json", default=None)
    parser.add_argument("--table-output", default=None)
    parser.add_argument("--clip-dir", default="checkpoints/clip-vit-base-patch32")
    parser.add_argument("--dino-dir", default="checkpoints/dino-vits16")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--max-items", type=int, default=None)
    return parser.parse_args()


def load_items(path: Path, max_items: int | None) -> list[dict]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    items = payload["items"] if isinstance(payload, dict) and "items" in payload else payload
    return items[:max_items] if max_items else items


def resolve_bench_path(raw_path: str, bench_root: Path) -> Path:
    path = Path(raw_path)
    if path.is_file():
        return path
    if path.parts and path.parts[0] == bench_root.name:
        return bench_root.parent / path
    return bench_root / path


def result_path_for(item: dict, result_dir: Path) -> Path:
    path = result_dir / f"{item['id']}.jpg"
    if path.is_file():
        return path
    return result_dir / f"{item['class_name']}_{item['id']}.jpg"


def mask_path_for(image_path: Path) -> Path:
    return image_path.with_name(f"{image_path.stem}_mask.jpg")


def align_mask_to_image(mask: Image.Image, image: Image.Image) -> Image.Image:
    return mask.resize(image.size, RESAMPLE_NEAREST) if mask.size != image.size else mask


def align_image_size(image: Image.Image, size: tuple[int, int]) -> Image.Image:
    return image.resize(size, RESAMPLE_BICUBIC) if image.size != size else image


def preprocess_lpips(image_pil: Image.Image, device: torch.device) -> torch.Tensor:
    transform = T.Compose([T.ToTensor(), T.Normalize(mean=[0.5] * 3, std=[0.5] * 3)])
    return transform(image_pil.convert("RGB")).unsqueeze(0).to(device)


def mask_white(image: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    if mask.shape[0] != 1:
        mask = mask[:1]
    blank = torch.ones_like(image[0])
    out = image.clone()
    for idx in range(out.shape[0]):
        out[idx] = torch.where(mask[0].gt(0), out[idx], blank)
    return out


def mask_reverse_white(image: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    if mask.shape[0] != 1:
        mask = mask[:1]
    blank = torch.ones_like(image[0])
    out = image.clone()
    for idx in range(out.shape[0]):
        out[idx] = torch.where(mask[0].gt(0), blank, out[idx])
    return out


def mask_crop_ori(image: torch.Tensor, mask: torch.Tensor, inter_size: bool = False, size_img: torch.Tensor | None = None) -> torch.Tensor:
    _, orig_h, orig_w = image.shape
    if mask.shape[0] != 1:
        mask = mask[:1]
    if mask.gt(0).any():
        y_indices, x_indices = torch.where(mask[0].gt(0))
        y_min, y_max = y_indices.min().item(), y_indices.max().item()
        x_min, x_max = x_indices.min().item(), x_indices.max().item()
        h_msk = min(orig_h, y_max - y_min + 60)
        w_msk = min(orig_w, x_max - x_min + 60)
        w_msk_4_h = h_msk * orig_w // orig_h
        new_w = min(max(w_msk, w_msk_4_h), orig_w)
        if new_w != orig_w:
            new_h = int(new_w * orig_h / orig_w)
            x_mi = max(int((x_min + x_max) / 2 - new_w / 2), 0)
            y_mi = max(int((y_min + y_max) / 2 - new_h / 2), 0)
            image = image[:, y_mi : y_mi + new_h, x_mi : x_mi + new_w]
    if inter_size:
        target_size = (orig_h, orig_w)
        if size_img is not None:
            target_size = (size_img.size(1), size_img.size(2))
        image = F.interpolate(image.unsqueeze(0), size=target_size, mode="bicubic", align_corners=False).squeeze(0)
    return image


def save_tensor_image(tensor: torch.Tensor, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if tensor.shape[0] != 3:
        tensor = tensor[:1].repeat(3, 1, 1)
    torchvision.utils.save_image(tensor, path)


def get_or_create_crop(image_path: Path, mask_path: Path, suffix: str, reverse: bool = False) -> Image.Image:
    crop_path = image_path.with_name(f"{image_path.stem}_{suffix}.jpg")
    if crop_path.is_file():
        return Image.open(crop_path).convert("RGB")
    image = Image.open(image_path).convert("RGB")
    mask = align_mask_to_image(Image.open(mask_path).convert("L"), image)
    image_tensor = T.ToTensor()(image)
    mask_tensor = T.ToTensor()(mask)
    masked = mask_reverse_white(image_tensor, mask_tensor) if reverse else mask_white(image_tensor, mask_tensor)
    if not reverse:
        masked = mask_crop_ori(masked, mask_tensor, inter_size=True, size_img=mask_tensor)
    save_tensor_image(masked, crop_path)
    return Image.open(crop_path).convert("RGB")


def macro_average(groups: dict[str, list[dict]], key: str) -> float:
    values = []
    for rows in groups.values():
        if rows:
            values.append(sum(float(row[key]) for row in rows) / len(rows))
    return sum(values) / len(values) if values else 0.0


def format_table(summary: dict, paper: dict | None = None) -> str:
    rows = [
        "| Method | Items | M-DINO ↑ | M-CLIP-I ↑ | LPIPS ↓ | SSIM ↑ | G-Acc. ↑ | CLIP-T ↑ |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    if paper:
        rows.append(
            f"| OmniTry paper | - | {paper['m_dino']:.4f} | {paper['m_clip_i']:.4f} | "
            f"{paper['lpips']:.4f} | {paper['ssim']:.4f} | {paper['g_acc']:.4f} | {paper['clip_t']:.4f} |"
        )
    rows.append(
        f"| Pretrained + Geometry | {summary['items']} | {summary['m_dino']:.4f} | "
        f"{summary['m_clip_i']:.4f} | {summary['lpips']:.4f} | {summary['ssim']:.4f} | "
        f"{summary['g_acc']:.4f} | {summary['clip_t']:.4f} |"
    )
    return "\n".join(rows) + "\n"


def main() -> int:
    args = parse_args()
    benchmark_file = ROOT / args.benchmark_file
    bench_root = ROOT / args.bench_root
    result_dir = ROOT / args.result_dir
    result_json = ROOT / args.result_json if args.result_json else result_dir / "official_result.json"
    detail_json = ROOT / args.detail_json if args.detail_json else result_dir / "official_result_detail.json"
    table_output = ROOT / args.table_output if args.table_output else result_dir / "official_metric_table.md"
    clip_dir = ROOT / args.clip_dir
    dino_dir = ROOT / args.dino_dir

    device = torch.device(args.device if torch.cuda.is_available() and args.device.startswith("cuda") else "cpu")
    clip_dir.mkdir(parents=True, exist_ok=True)
    dino_dir.mkdir(parents=True, exist_ok=True)
    snapshot_download(repo_id="openai/clip-vit-base-patch32", local_dir=str(clip_dir))
    snapshot_download(repo_id="facebook/dino-vits16", local_dir=str(dino_dir))

    clip_model = CLIPModel.from_pretrained(str(clip_dir)).requires_grad_(False).to(device).eval()
    clip_processor = CLIPProcessor.from_pretrained(str(clip_dir))
    dino_model = ViTModel.from_pretrained(str(dino_dir)).requires_grad_(False).to(device).eval()
    dino_processor = ViTImageProcessor.from_pretrained(str(dino_dir))
    lpips_model = lpips.LPIPS(net="alex", version="0.1").to(device).eval()
    ssim_model = StructuralSimilarityIndexMeasure(data_range=1.0).to(device).eval()

    items = load_items(benchmark_file, args.max_items)
    to_tensor = T.ToTensor()
    details = []
    groups: dict[str, list[dict]] = defaultdict(list)
    skip_infos = []
    result_mask_none = 0

    for item in tqdm(items, desc="official metrics"):
        item_id = item["id"]
        garment_class = item["garment_class"]
        big_class = item["class_name"].split("_")[0]
        model_path = resolve_bench_path(item["person"]["img_path"], bench_root)
        object_path = resolve_bench_path(item["object"]["img_path"], bench_root)
        tryon_path = result_path_for(item, result_dir)
        result_mask_path = mask_path_for(tryon_path)
        object_mask_path = mask_path_for(object_path)
        try:
            if not tryon_path.is_file():
                raise FileNotFoundError(tryon_path)
            if not result_mask_path.is_file():
                raise FileNotFoundError(result_mask_path)
            if not object_mask_path.is_file():
                raise FileNotFoundError(object_mask_path)

            model_img = Image.open(model_path).convert("RGB")
            tryon_img = Image.open(tryon_path).convert("RGB")
            result_mask = align_mask_to_image(Image.open(result_mask_path).convert("L"), tryon_img)
            mask_tensor = to_tensor(result_mask)
            if mask_tensor.eq(0).all():
                result_mask_none += 1

            result_crop = get_or_create_crop(tryon_path, result_mask_path, "crop_white", reverse=False)
            object_crop = get_or_create_crop(object_path, object_mask_path, "crop_white", reverse=False)

            with torch.no_grad():
                dino_gen = dino_model(**dino_processor(images=result_crop, return_tensors="pt", padding=True).to(device)).last_hidden_state[:, 0]
                dino_obj = dino_model(**dino_processor(images=object_crop, return_tensors="pt", padding=True).to(device)).last_hidden_state[:, 0]
            dino_gen = dino_gen / dino_gen.norm(p=2, dim=-1, keepdim=True)
            dino_obj = dino_obj / dino_obj.norm(p=2, dim=-1, keepdim=True)
            m_dino = F.cosine_similarity(dino_gen, dino_obj).item()

            with torch.no_grad():
                clip_gen = clip_model.get_image_features(**clip_processor(images=result_crop, return_tensors="pt", padding=True).to(device))
                clip_obj = clip_model.get_image_features(**clip_processor(images=object_crop, return_tensors="pt", padding=True).to(device))
            clip_gen = clip_gen / clip_gen.norm(p=2, dim=-1, keepdim=True)
            clip_obj = clip_obj / clip_obj.norm(p=2, dim=-1, keepdim=True)
            m_clip_i = torch.matmul(clip_gen, clip_obj.T).item()

            text_prompt = [item.get("gt", {}).get("caption") or f"{garment_class} {PROMPT_POS.get(garment_class, 'on the person')} of the model"]
            with torch.no_grad():
                clip_inputs = clip_processor(text=text_prompt, images=tryon_img, return_tensors="pt", padding=True).to(device)
                if clip_inputs["input_ids"].size(-1) >= 77 and item.get("gt", {}).get("caption"):
                    text_prompt = [item["gt"]["caption"].split(". ")[0]]
                    clip_inputs = clip_processor(text=text_prompt, images=tryon_img, return_tensors="pt", padding=True).to(device)
                clip_text_image = clip_model(**clip_inputs)
                clip_t_logit = clip_text_image.logits_per_image.item()
                clip_t_image = clip_text_image.image_embeds / clip_text_image.image_embeds.norm(p=2, dim=-1, keepdim=True)
                clip_t_text = clip_text_image.text_embeds / clip_text_image.text_embeds.norm(p=2, dim=-1, keepdim=True)
                clip_t = torch.matmul(clip_t_image, clip_t_text.T).item()

            result_residue = get_or_create_crop(tryon_path, result_mask_path, "residue_white", reverse=True)
            result_residue = align_image_size(result_residue, tryon_img.size)
            model_residue_path = result_dir / f"{item['class_name']}_{item_id}_model_residue.jpg"
            if not model_residue_path.is_file():
                model_tensor = T.Compose([T.Resize((tryon_img.height, tryon_img.width)), T.ToTensor()])(model_img)
                model_residue_tensor = mask_reverse_white(model_tensor, mask_tensor)
                save_tensor_image(model_residue_tensor, model_residue_path)
            model_residue = align_image_size(Image.open(model_residue_path).convert("RGB"), tryon_img.size)

            with torch.no_grad():
                lpips_value = lpips_model(preprocess_lpips(model_residue, device), preprocess_lpips(result_residue, device)).item()
            result_residue_tensor = to_tensor(result_residue).unsqueeze(0).to(device)
            model_residue_tensor = to_tensor(model_residue).unsqueeze(0).to(device)
            with torch.no_grad():
                ssim_value = ssim_model(model_residue_tensor, result_residue_tensor).item()

            row = {
                "id": item_id,
                "class_name": item["class_name"],
                "garment_class": garment_class,
                "gen_tryon": {"img_path": str(tryon_path)},
                "M-DINO": m_dino,
                "M-CLIP-I": m_clip_i,
                "CLIP-T": clip_t,
                "CLIP-T-logit": clip_t_logit,
                "LPIPS": lpips_value,
                "SSIM": ssim_value,
                "mask_empty": bool(mask_tensor.eq(0).all()),
            }
            details.append(row)
            groups[big_class].append(row)
        except Exception as exc:  # noqa: BLE001
            skip_infos.append(
                {
                    "id": item_id,
                    "class_name": item.get("class_name"),
                    "person_path": str(model_path),
                    "object_path": str(object_path),
                    "tryon_path": str(tryon_path),
                    "error": str(exc),
                }
            )

    class_summary = {}
    for group, rows in sorted(groups.items()):
        class_summary[group] = {
            "count": len(rows),
            "M-DINO": sum(row["M-DINO"] for row in rows) / len(rows),
            "M-CLIP-I": sum(row["M-CLIP-I"] for row in rows) / len(rows),
            "LPIPS": sum(row["LPIPS"] for row in rows) / len(rows),
            "SSIM": sum(row["SSIM"] for row in rows) / len(rows),
            "CLIP-T": sum(row["CLIP-T"] for row in rows) / len(rows),
        }

    summary = {
        "benchmark_file": str(benchmark_file),
        "result_dir": str(result_dir),
        "items": len(details),
        "expected_items": len(items),
        "skipped": len(skip_infos),
        "m_dino": macro_average(groups, "M-DINO"),
        "m_clip_i": macro_average(groups, "M-CLIP-I"),
        "lpips": macro_average(groups, "LPIPS"),
        "ssim": macro_average(groups, "SSIM"),
        "g_acc": (len(items) - result_mask_none) / float(len(items)) if items else 0.0,
        "clip_t": macro_average(groups, "CLIP-T"),
        "classes": class_summary,
        "skip_infos": skip_infos,
        "complete": len(details) == len(items) and not skip_infos,
    }

    result_json.parent.mkdir(parents=True, exist_ok=True)
    detail_json.parent.mkdir(parents=True, exist_ok=True)
    table_output.parent.mkdir(parents=True, exist_ok=True)
    result_json.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    detail_json.write_text(json.dumps(details, indent=2), encoding="utf-8")
    paper = {
        "m_dino": 0.6160,
        "m_clip_i": 0.8327,
        "lpips": 0.0542,
        "ssim": 0.9333,
        "g_acc": 0.9972,
        "clip_t": 0.2831,
    }
    table = "# Official OmniTry-Bench Metrics\n\n" + format_table(summary, paper=paper)
    table += (
        "\nPaper numbers are from arXiv:2508.13632 Table 1. "
        f"Our row is computed on {summary['expected_items']}-case OmniTry-Bench manifest. "
        "CLIP-T is normalized image-text cosine for paper-scale comparison; raw CLIP logits are saved in the detail JSON.\n"
    )
    if skip_infos:
        table += f"\nWarning: skipped {len(skip_infos)} items. See `{result_json}`.\n"
    table_output.write_text(table, encoding="utf-8")
    print(json.dumps(summary, indent=2))
    print(table)
    return 0 if summary["complete"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
