#!/usr/bin/env python
import argparse
import json
import os
import random
import sys
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

import numpy as np
import torch
import torch.nn.functional as F
import torchvision.transforms as T
from accelerate import Accelerator
from accelerate.utils import set_seed
from omegaconf import OmegaConf
from PIL import Image, ImageFile
from torch.utils.data import DataLoader, Dataset
from tqdm.auto import tqdm

ImageFile.LOAD_TRUNCATED_IMAGES = True

from omnitry.enhance import build_enhanced_prompt
from omnitry.enhance.flux_lora import (
    add_omnitry_lora_adapters,
    load_lora_safetensors,
    patch_dual_stream_lora,
    save_lora_safetensors,
    set_only_lora_trainable,
)
from omnitry.enhance.taxonomy import normalize_class_name
from omnitry.models.transformer_flux import FluxTransformer2DModel
from omnitry.pipelines.pipeline_flux import calculate_shift, retrieve_timesteps
from omnitry.pipelines.pipeline_flux_fill import FluxFillPipeline


TARGET_KEYS = ("target_path", "gt_path", "tryon_path", "result_path", "image_path")


def parse_args():
    parser = argparse.ArgumentParser(description="Fine-tune OmniTry dual-stream FLUX LoRA adapters.")
    parser.add_argument("--config", default="configs/omnitry_v1_unified.yaml")
    parser.add_argument("--manifest", default="data/hard_cases/omnitry_full_local_hard_cases.json")
    parser.add_argument("--output", default="checkpoints/enhance/omnitry_geo_lora.safetensors")
    parser.add_argument("--metrics-output", default="outputs/enhance/geo_lora_train_metrics.json")
    parser.add_argument("--model-root", default=None)
    parser.add_argument("--init-lora", default=None, help="Existing OmniTry LoRA to continue from.")
    parser.add_argument("--resolution", type=int, default=512)
    parser.add_argument("--max-train-steps", type=int, default=1000)
    parser.add_argument("--train-batch-size", type=int, default=1)
    parser.add_argument("--gradient-accumulation-steps", type=int, default=4)
    parser.add_argument("--learning-rate", type=float, default=1e-5)
    parser.add_argument("--weight-decay", type=float, default=0.0)
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--num-timesteps", type=int, default=1000)
    parser.add_argument("--guidance-scale", type=float, default=30.0)
    parser.add_argument("--lora-rank", type=int, default=None)
    parser.add_argument("--lora-alpha", type=int, default=None)
    parser.add_argument("--garment-loss-weight", type=float, default=0.25)
    parser.add_argument("--save-every-steps", type=int, default=100)
    parser.add_argument("--max-items", type=int, default=None)
    parser.add_argument("--seed", type=int, default=123)
    parser.add_argument("--allow-person-target", action="store_true")
    parser.add_argument("--validate-data-only", action="store_true")
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


def resolve_path(raw_path, manifest_dir):
    if not raw_path:
        return None
    path = Path(raw_path)
    if path.is_absolute():
        return path
    if path.is_file():
        return path
    candidate = manifest_dir / path
    if candidate.is_file():
        return candidate
    return path


def target_path_for(item, manifest_dir, allow_person_target=False):
    for key in TARGET_KEYS:
        path = resolve_path(item.get(key), manifest_dir)
        if path and path.is_file():
            return path, False
    if allow_person_target:
        path = resolve_path(item.get("person_path"), manifest_dir)
        if path and path.is_file():
            return path, True
    return None, False


class TryOnLoraDataset(Dataset):
    def __init__(self, manifest, config, resolution=512, max_items=None, allow_person_target=False):
        self.manifest = Path(manifest)
        payload = load_json(self.manifest)
        raw_items = payload["items"] if isinstance(payload, dict) and "items" in payload else payload
        if max_items is not None:
            raw_items = raw_items[:max_items]

        self.items = []
        missing_target = 0
        for item in raw_items:
            copied = dict(item)
            person_path = resolve_path(copied.get("person_path"), self.manifest.parent)
            object_path = resolve_path(copied.get("object_path"), self.manifest.parent)
            target_path, target_is_person = target_path_for(copied, self.manifest.parent, allow_person_target)
            if not person_path or not object_path or not person_path.is_file() or not object_path.is_file():
                continue
            if not target_path:
                missing_target += 1
                continue
            copied["person_path"] = str(person_path)
            copied["object_path"] = str(object_path)
            copied["target_path"] = str(target_path)
            copied["target_is_person_fallback"] = target_is_person
            self.items.append(copied)

        if not self.items:
            raise ValueError(
                "No trainable paired items found. Manifest entries need person_path, object_path, "
                "and target_path/gt_path. Pass --allow-person-target only for reconstruction smoke tests."
            )

        self.resolution = resolution
        self.object_map = dict(config.object_map)
        self.to_tensor = T.ToTensor()
        self.missing_target = missing_target

    def __len__(self):
        return len(self.items)

    def _fit_canvas(self, image):
        image = image.convert("RGB")
        image.thumbnail((self.resolution, self.resolution), Image.Resampling.LANCZOS)
        canvas = Image.new("RGB", (self.resolution, self.resolution), (255, 255, 255))
        left = (self.resolution - image.width) // 2
        top = (self.resolution - image.height) // 2
        canvas.paste(image, (left, top))
        return canvas

    def _object_canvas(self, image):
        image = image.convert("RGB")
        ratio = min(self.resolution / image.width, self.resolution / image.height)
        size = (max(1, int(image.width * ratio)), max(1, int(image.height * ratio)))
        image = image.resize(size, Image.Resampling.LANCZOS)
        canvas = Image.new("RGB", (self.resolution, self.resolution), (255, 255, 255))
        left = (self.resolution - image.width) // 2
        top = (self.resolution - image.height) // 2
        canvas.paste(image, (left, top))
        return canvas

    def __getitem__(self, index):
        item = self.items[index]
        category = normalize_class_name(item.get("category") or item.get("garment_class", ""))
        base_prompt = self.object_map.get(category, f"trying on {category}")
        optional_prompt = item.get("gt", {}).get("caption_cate") or item.get("gt", {}).get("caption") or ""
        prompt = build_enhanced_prompt(base_prompt, category, optional_prompt, enabled=True)

        person = self._fit_canvas(Image.open(item["person_path"]))
        target = self._fit_canvas(Image.open(item["target_path"]))
        object_canvas = self._object_canvas(Image.open(item["object_path"]))

        return {
            "person": self.to_tensor(person),
            "object": self.to_tensor(object_canvas),
            "target": self.to_tensor(target),
            "prompt": prompt,
            "id": item.get("id", str(index)),
            "category": category,
            "target_is_person_fallback": item.get("target_is_person_fallback", False),
        }


def collate_one(examples):
    if len(examples) != 1:
        raise ValueError("train_geo_lora.py currently requires --train-batch-size 1.")
    return examples[0]


def validate_data(args, config):
    dataset = TryOnLoraDataset(
        args.manifest,
        config,
        resolution=args.resolution,
        max_items=args.max_items,
        allow_person_target=args.allow_person_target,
    )
    fallback_count = sum(1 for item in dataset.items if item.get("target_is_person_fallback"))
    print(f"Trainable items: {len(dataset)}")
    print(f"Missing target entries skipped: {dataset.missing_target}")
    print(f"Person-target fallback items: {fallback_count}")
    print(f"Resolution: {args.resolution}")
    print(f"Example id: {dataset.items[0].get('id')}")


def load_training_pipeline(args, config, accelerator):
    model_root = args.model_root or config.model_root
    init_lora = args.init_lora or config.lora_path
    rank = args.lora_rank or config.lora_rank
    alpha = args.lora_alpha or config.lora_alpha
    dtype = torch.bfloat16 if torch.cuda.is_available() else torch.float32

    transformer = FluxTransformer2DModel.from_pretrained(f"{model_root}/transformer").to(dtype=dtype)
    transformer.requires_grad_(False)
    add_omnitry_lora_adapters(transformer, rank=rank, alpha=alpha)
    load_lora_safetensors(transformer, init_lora)
    patch_dual_stream_lora(transformer)
    set_only_lora_trainable(transformer)

    if hasattr(transformer, "enable_gradient_checkpointing"):
        transformer.enable_gradient_checkpointing()
    else:
        transformer.gradient_checkpointing = True

    pipe = FluxFillPipeline.from_pretrained(model_root, transformer=transformer, torch_dtype=dtype)
    pipe.to(accelerator.device)
    pipe.vae.enable_tiling()
    pipe.vae.requires_grad_(False).eval()
    pipe.text_encoder.requires_grad_(False).eval()
    pipe.text_encoder_2.requires_grad_(False).eval()
    pipe.transformer.train()
    return pipe, transformer, dtype


def trainable_parameters(model):
    return [param for param in model.parameters() if param.requires_grad]


def transformer_config(transformer):
    return transformer.module.config if hasattr(transformer, "module") else transformer.config


def build_metrics(args, dataset, global_step, history, checkpoint):
    return {
        "checkpoint": checkpoint,
        "manifest": args.manifest,
        "items": len(dataset),
        "resolution": args.resolution,
        "steps": global_step,
        "max_train_steps": args.max_train_steps,
        "gradient_accumulation_steps": args.gradient_accumulation_steps,
        "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
        "history": history,
    }


def pipe_device(pipe):
    return pipe._execution_device


@torch.no_grad()
def encode_images(pipe, images, dtype):
    images = images.to(device=pipe_device(pipe), dtype=dtype) * 2.0 - 1.0
    latents = pipe.vae.encode(images).latent_dist.sample()
    return (latents - pipe.vae.config.shift_factor) * pipe.vae.config.scaling_factor


@torch.no_grad()
def prepare_condition(pipe, person, object_image, dtype):
    device = pipe_device(pipe)
    img_cond = torch.stack([person, object_image], dim=0).to(device=device, dtype=dtype)
    mask = torch.zeros_like(img_cond)
    img_cond = pipe.image_processor.preprocess(img_cond, height=person.shape[-2], width=person.shape[-1])
    mask = pipe.mask_processor.preprocess(mask, height=person.shape[-2], width=person.shape[-1])
    img_cond = img_cond.to(device=device, dtype=dtype)
    mask = mask.to(device=device, dtype=dtype)
    masked_image = img_cond * (1 - mask)
    mask, masked_image_latents = pipe.prepare_mask_latents(
        mask,
        masked_image,
        batch_size=2,
        num_channels_latents=pipe.vae.config.latent_channels,
        num_images_per_prompt=1,
        height=person.shape[-2],
        width=person.shape[-1],
        dtype=dtype,
        device=device,
        generator=None,
    )
    return torch.cat((masked_image_latents, mask), dim=-1)


def training_step(pipe, transformer, batch, args, dtype):
    device = pipe_device(pipe)
    target_streams = torch.stack([batch["target"], batch["object"]], dim=0)
    target_latents = encode_images(pipe, target_streams, dtype)
    latent_bsz, latent_channels, latent_h, latent_w = target_latents.shape

    sigmas = np.linspace(1.0, 1 / args.num_timesteps, args.num_timesteps)
    image_seq_len = (latent_h // 2) * (latent_w // 2)
    mu = calculate_shift(
        image_seq_len,
        pipe.scheduler.config.base_image_seq_len,
        pipe.scheduler.config.max_image_seq_len,
        pipe.scheduler.config.base_shift,
        pipe.scheduler.config.max_shift,
    )
    retrieve_timesteps(pipe.scheduler, args.num_timesteps, device, sigmas=sigmas, mu=mu)
    timestep_index = torch.randint(0, len(pipe.scheduler.timesteps), (1,), device=device)
    timesteps = pipe.scheduler.timesteps[timestep_index].expand(latent_bsz)
    noise = torch.randn_like(target_latents)
    noisy_latents = pipe.scheduler.scale_noise(target_latents, timesteps, noise)
    target_velocity = noise - target_latents

    noisy_latents = pipe._pack_latents(noisy_latents, latent_bsz, latent_channels, latent_h, latent_w)
    target_velocity = pipe._pack_latents(target_velocity, latent_bsz, latent_channels, latent_h, latent_w)
    condition_latents = prepare_condition(pipe, batch["person"], batch["object"], dtype)
    latent_image_ids = pipe._prepare_latent_image_ids(
        latent_bsz,
        latent_h,
        latent_w,
        device,
        dtype,
    )

    with torch.no_grad():
        prompt_embeds, pooled_prompt_embeds, text_ids = pipe.encode_prompt(
            prompt=[batch["prompt"], batch["prompt"]],
            prompt_2=None,
            device=device,
            num_images_per_prompt=1,
            max_sequence_length=512,
        )

    guidance = None
    if transformer_config(transformer).guidance_embeds:
        guidance = torch.full([latent_bsz], args.guidance_scale, device=device, dtype=torch.float32)

    model_pred = transformer(
        hidden_states=torch.cat((noisy_latents, condition_latents), dim=-1),
        timestep=timesteps.to(dtype) / 1000,
        guidance=guidance,
        pooled_projections=pooled_prompt_embeds,
        encoder_hidden_states=prompt_embeds,
        txt_ids=text_ids,
        img_ids=latent_image_ids,
        return_dict=False,
    )[0]

    vtryon_loss = F.mse_loss(model_pred[:1].float(), target_velocity[:1].float())
    garment_loss = F.mse_loss(model_pred[1:].float(), target_velocity[1:].float())
    return vtryon_loss + args.garment_loss_weight * garment_loss


def main():
    args = parse_args()
    if args.resolution % 16 != 0:
        raise ValueError("--resolution must be divisible by 16.")
    if args.train_batch_size != 1:
        raise ValueError("The custom OmniTry attention path requires --train-batch-size 1.")

    config = OmegaConf.load(args.config)
    set_seed(args.seed)
    random.seed(args.seed)
    np.random.seed(args.seed)

    if args.validate_data_only:
        validate_data(args, config)
        return 0

    accelerator = Accelerator(gradient_accumulation_steps=args.gradient_accumulation_steps)
    dataset = TryOnLoraDataset(
        args.manifest,
        config,
        resolution=args.resolution,
        max_items=args.max_items,
        allow_person_target=args.allow_person_target,
    )
    dataloader = DataLoader(
        dataset,
        batch_size=args.train_batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        collate_fn=collate_one,
        pin_memory=torch.cuda.is_available(),
    )

    pipe, transformer, dtype = load_training_pipeline(args, config, accelerator)
    optimizer = torch.optim.AdamW(trainable_parameters(transformer), lr=args.learning_rate, weight_decay=args.weight_decay)
    transformer, optimizer, dataloader = accelerator.prepare(transformer, optimizer, dataloader)
    pipe.transformer = transformer

    progress = tqdm(
        total=args.max_train_steps,
        disable=not accelerator.is_main_process,
        desc="Fine-tuning OmniTry LoRA",
    )
    history = []
    global_step = 0
    while global_step < args.max_train_steps:
        for batch in dataloader:
            with accelerator.accumulate(transformer):
                with accelerator.autocast():
                    loss = training_step(pipe, transformer, batch, args, dtype)
                accelerator.backward(loss)
                if accelerator.sync_gradients:
                    accelerator.clip_grad_norm_(trainable_parameters(transformer), 1.0)
                optimizer.step()
                optimizer.zero_grad(set_to_none=True)

            if accelerator.sync_gradients:
                global_step += 1
                mean_loss = accelerator.reduce(loss.detach(), reduction="mean").item()
                if accelerator.is_main_process:
                    progress.update(1)
                    progress.set_postfix(loss=f"{mean_loss:.4f}")
                    if global_step == 1 or global_step % 10 == 0:
                        history.append({"step": global_step, "loss": round(float(mean_loss), 6)})
                    if args.save_every_steps > 0 and global_step % args.save_every_steps == 0:
                        unwrapped = accelerator.unwrap_model(transformer)
                        save_lora_safetensors(unwrapped, args.output)
                        write_json(
                            args.metrics_output,
                            build_metrics(args, dataset, global_step, history, args.output),
                        )
                        progress.write(f"Saved interim LoRA at step {global_step} -> {args.output}")
                if global_step >= args.max_train_steps:
                    break

    accelerator.wait_for_everyone()
    if accelerator.is_main_process:
        progress.close()
        unwrapped = accelerator.unwrap_model(transformer)
        save_lora_safetensors(unwrapped, args.output)
        write_json(args.metrics_output, build_metrics(args, dataset, global_step, history, args.output))
        print(f"Saved LoRA -> {args.output}")
    accelerator.end_training()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
