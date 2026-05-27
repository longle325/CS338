from pathlib import Path

import peft
import torch
from peft import LoraConfig
from safetensors import safe_open
from safetensors.torch import save_file


FLUX_LORA_TARGET_MODULES = [
    "x_embedder",
    "attn.to_k",
    "attn.to_q",
    "attn.to_v",
    "attn.to_out.0",
    "attn.add_k_proj",
    "attn.add_q_proj",
    "attn.add_v_proj",
    "attn.to_add_out",
    "ff.net.0.proj",
    "ff.net.2",
    "ff_context.net.0.proj",
    "ff_context.net.2",
    "norm1_context.linear",
    "norm1.linear",
    "norm.linear",
    "proj_mlp",
    "proj_out",
]


def create_hacked_forward(module):
    def lora_forward(self, active_adapter, x, *args, **kwargs):
        result = self.base_layer(x, *args, **kwargs)
        if active_adapter is not None:
            lora_a = self.lora_A[active_adapter]
            lora_b = self.lora_B[active_adapter]
            dropout = self.lora_dropout[active_adapter]
            scaling = self.scaling[active_adapter]
            x = x.to(lora_a.weight.dtype)
            result = result + lora_b(lora_a(dropout(x))) * scaling
        return result

    def hacked_lora_forward(self, x, *args, **kwargs):
        return torch.cat(
            (
                lora_forward(self, "vtryon_lora", x[:1], *args, **kwargs),
                lora_forward(self, "garment_lora", x[1:], *args, **kwargs),
            ),
            dim=0,
        )

    return hacked_lora_forward.__get__(module, type(module))


def add_omnitry_lora_adapters(transformer, rank=16, alpha=16):
    config = LoraConfig(
        r=rank,
        lora_alpha=alpha,
        init_lora_weights="gaussian",
        target_modules=FLUX_LORA_TARGET_MODULES,
    )
    transformer.add_adapter(config, adapter_name="vtryon_lora")
    transformer.add_adapter(config, adapter_name="garment_lora")
    return transformer


def patch_dual_stream_lora(transformer):
    for _, module in transformer.named_modules():
        if isinstance(module, peft.tuners.lora.layer.Linear):
            module.forward = create_hacked_forward(module)


def load_lora_safetensors(transformer, path):
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(f"LoRA checkpoint not found: {path}")
    with safe_open(path, framework="pt") as handle:
        weights = {key: handle.get_tensor(key) for key in handle.keys()}
    return transformer.load_state_dict(weights, strict=False)


def set_only_lora_trainable(transformer):
    for name, param in transformer.named_parameters():
        param.requires_grad = "lora_" in name


def lora_state_dict(transformer):
    return {
        key: value.detach().cpu().contiguous()
        for key, value in transformer.state_dict().items()
        if "lora_" in key
    }


def save_lora_safetensors(transformer, path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    save_file(lora_state_dict(transformer), str(path))
