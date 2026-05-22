import gradio as gr
import torch
import random
import torchvision.transforms as T
import math
import peft
from pathlib import Path
from typing import List
from PIL import Image
from peft import LoraConfig
from safetensors import safe_open
from omegaconf import OmegaConf
import os
os.environ.setdefault("GRADIO_TEMP_DIR", ".gradio")

from omnitry.models.transformer_flux import FluxTransformer2DModel
from omnitry.pipelines.pipeline_flux_fill import FluxFillPipeline
from omnitry.enhance import (
    CandidateResult,
    build_enhanced_prompt,
    confidence_label,
    format_diagnostics,
    score_candidate,
)


def _env_int(name, default):
    try:
        return int(os.environ.get(name, default))
    except (TypeError, ValueError):
        return default


device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
weight_dtype = torch.bfloat16 if torch.cuda.is_available() else torch.float32
args = OmegaConf.load('configs/omnitry_v1_unified.yaml')

MAX_AREA = max(16 * 16, _env_int("OMNITRY_MAX_AREA", 1024 * 1024))
MAX_CANDIDATES = max(1, _env_int("OMNITRY_MAX_CANDIDATES", 4))
SEED_STRIDE = 9973
transformer = None
pipeline = None


def create_hacked_forward(module):

    def lora_forward(self, active_adapter, x, *args, **kwargs):
        result = self.base_layer(x, *args, **kwargs)
        if active_adapter is not None:
            torch_result_dtype = result.dtype
            lora_A = self.lora_A[active_adapter]
            lora_B = self.lora_B[active_adapter]
            dropout = self.lora_dropout[active_adapter]
            scaling = self.scaling[active_adapter]
            x = x.to(lora_A.weight.dtype)
            result = result + lora_B(lora_A(dropout(x))) * scaling
        return result
    
    def hacked_lora_forward(self, x, *args, **kwargs):
        return torch.cat((
            lora_forward(self, 'vtryon_lora', x[:1], *args, **kwargs),
            lora_forward(self, 'garment_lora', x[1:], *args, **kwargs),
        ), dim=0)
    
    return hacked_lora_forward.__get__(module, type(module))


def validate_checkpoint_paths():
    missing = []
    model_root = Path(args.model_root)
    transformer_root = model_root / 'transformer'
    lora_path = Path(args.lora_path)

    if not model_root.exists():
        missing.append(f'model_root: {model_root}')
    if not transformer_root.exists():
        missing.append(f'transformer: {transformer_root}')
    if not lora_path.is_file():
        missing.append(f'lora_path: {lora_path}')

    if missing:
        message = (
            'Missing OmniTry checkpoints:\n'
            + '\n'.join(f'- {item}' for item in missing)
            + '\n\nRun `bash scripts/setup_omnitry.sh` with `HF_TOKEN` set, or download the checkpoints manually.'
        )
        raise gr.Error(message)


def load_pipeline():
    global transformer, pipeline

    if pipeline is not None:
        return pipeline

    validate_checkpoint_paths()

    transformer = FluxTransformer2DModel.from_pretrained(f'{args.model_root}/transformer').requires_grad_(False).to(dtype=weight_dtype)
    pipeline = FluxFillPipeline.from_pretrained(args.model_root, transformer=transformer.eval(), torch_dtype=weight_dtype)

    # VRAM saving, comment the following lines if you have sufficient memory.
    if torch.cuda.is_available():
        pipeline.enable_model_cpu_offload()
    else:
        pipeline.to(device)
    pipeline.vae.enable_tiling()

    lora_config = LoraConfig(
        r=args.lora_rank,
        lora_alpha=args.lora_alpha,
        init_lora_weights="gaussian",
        target_modules=[
            'x_embedder',
            'attn.to_k', 'attn.to_q', 'attn.to_v', 'attn.to_out.0',
            'attn.add_k_proj', 'attn.add_q_proj', 'attn.add_v_proj', 'attn.to_add_out',
            'ff.net.0.proj', 'ff.net.2', 'ff_context.net.0.proj', 'ff_context.net.2',
            'norm1_context.linear', 'norm1.linear', 'norm.linear', 'proj_mlp', 'proj_out'
        ]
    )
    transformer.add_adapter(lora_config, adapter_name='vtryon_lora')
    transformer.add_adapter(lora_config, adapter_name='garment_lora')

    with safe_open(args.lora_path, framework="pt") as f:
        lora_weights = {k: f.get_tensor(k) for k in f.keys()}
        transformer.load_state_dict(lora_weights, strict=False)

    for _, module in transformer.named_modules():
        if isinstance(module, peft.tuners.lora.layer.Linear):
            module.forward = create_hacked_forward(module)

    return pipeline


def seed_everything(seed=0):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)


def _clamp_int(value, default, minimum, maximum):
    try:
        value = int(value)
    except (TypeError, ValueError):
        value = default
    return max(minimum, min(maximum, value))


def _clamp_float(value, default, minimum, maximum):
    try:
        value = float(value)
    except (TypeError, ValueError):
        value = default
    return max(minimum, min(maximum, value))


def _as_rgb(image, name):
    if image is None:
        raise gr.Error(f'{name} is required.')
    if getattr(image, 'mode', None) == 'RGBA':
        background = Image.new('RGBA', image.size, (255, 255, 255, 255))
        return Image.alpha_composite(background, image).convert('RGB')
    return image.convert('RGB')


def _target_size(width, height):
    if width <= 0 or height <= 0:
        raise gr.Error('Input images must have positive width and height.')
    ratio = min(1.0, math.sqrt(MAX_AREA / (width * height)))
    target_width = max(16, int(width * ratio) // 16 * 16)
    target_height = max(16, int(height * ratio) // 16 * 16)
    return target_width, target_height


def build_prompt(object_class, optional_prompt='', enhance_mode='Enhanced'):
    if object_class not in args.object_map:
        raise gr.Error(f'Unknown object class: {object_class}')

    return build_enhanced_prompt(
        args.object_map[object_class],
        object_class,
        optional_prompt,
        enabled=enhance_mode == 'Enhanced',
    )


def prepare_condition_tensors(person_image, object_image):
    person_image = _as_rgb(person_image, 'Person image')
    object_image = _as_rgb(object_image, 'Object image')

    tW, tH = _target_size(person_image.width, person_image.height)
    transform = T.Compose([
        T.Resize((tH, tW)),
        T.ToTensor(),
    ])
    person_tensor = transform(person_image)

    ratio = min(tW / object_image.width, tH / object_image.height)
    transform = T.Compose([
        T.Resize((int(object_image.height * ratio), int(object_image.width * ratio))),
        T.ToTensor(),
    ])
    object_image_padded = torch.ones_like(person_tensor)
    object_tensor = transform(object_image)
    new_h, new_w = object_tensor.shape[1], object_tensor.shape[2]
    min_x = (tW - new_w) // 2
    min_y = (tH - new_h) // 2
    object_image_padded[:, min_y: min_y + new_h, min_x: min_x + new_w] = object_tensor

    return person_image, object_image, person_tensor, object_image_padded, tW, tH


def _run_candidate(person_tensor, object_image_padded, prompt, steps, guidance_scale, seed, tW, tH):
    seed_everything(seed)
    pipe = load_pipeline()
    prompts = [prompt] * 2
    img_cond = torch.stack([person_tensor, object_image_padded]).to(dtype=weight_dtype, device=device)
    mask = torch.zeros_like(img_cond).to(img_cond)
    generator = torch.Generator(device=device).manual_seed(seed)

    with torch.no_grad():
        img = pipe(
            prompt=prompts,
            height=tH,
            width=tW,
            img_cond=img_cond,
            mask=mask,
            guidance_scale=guidance_scale,
            num_inference_steps=steps,
            generator=generator,
        ).images[0]

    return img


def _resolve_seed(seed):
    try:
        seed = int(seed)
    except (TypeError, ValueError):
        seed = -1
    if seed < 0:
        seed = random.randint(0, 2**32 - 1)
    return seed % (2**32)


def _candidate_seeds(seed, count):
    return [(seed + i * SEED_STRIDE) % (2**32) for i in range(count)]


def generate(
    person_image,
    object_image,
    object_class,
    optional_prompt='',
    steps=20,
    guidance_scale=30,
    seed=-1,
    candidate_count=1,
    enhance_mode='Enhanced',
    progress=gr.Progress(track_tqdm=True),
):
    # Backward compatibility for the old positional signature:
    # generate(person, object, class, steps, guidance_scale, seed)
    if not isinstance(optional_prompt, str):
        old_steps, old_guidance_scale, old_seed = optional_prompt, steps, guidance_scale
        optional_prompt = ''
        steps = old_steps
        guidance_scale = old_guidance_scale
        seed = old_seed

    steps = _clamp_int(steps, 20, 1, 50)
    guidance_scale = _clamp_float(guidance_scale, 30, 1, 50)
    candidate_count = _clamp_int(candidate_count, 1, 1, MAX_CANDIDATES)
    enhance_mode = enhance_mode if enhance_mode in {'Enhanced', 'Baseline'} else 'Enhanced'
    effective_candidate_count = candidate_count if enhance_mode == 'Enhanced' else 1
    seed = _resolve_seed(seed)
    prompt = build_prompt(object_class, optional_prompt, enhance_mode)

    person_image, object_image, person_tensor, object_image_padded, tW, tH = prepare_condition_tensors(person_image, object_image)
    candidates: List[CandidateResult] = []

    for candidate_seed in progress.tqdm(_candidate_seeds(seed, effective_candidate_count), desc='Generating candidates'):
        image = _run_candidate(person_tensor, object_image_padded, prompt, steps, guidance_scale, candidate_seed, tW, tH)
        total, object_score, person_score, artifact_score = score_candidate(image, person_image, object_image, object_class)
        candidates.append(CandidateResult(image, candidate_seed, total, object_score, person_score, artifact_score))

    best_index = max(range(len(candidates)), key=lambda index: candidates[index].score)
    gallery = [
        (
            item.image,
            f'seed={item.seed} score={item.score:.3f} confidence={confidence_label(item.score)}',
        )
        for item in sorted(candidates, key=lambda candidate: candidate.score, reverse=True)
    ]

    return candidates[best_index].image, gallery, format_diagnostics(prompt, candidates, best_index, mode=enhance_mode)


if __name__ == '__main__':

    with gr.Blocks() as demo:
        gr.Markdown('# Demo of OmniTry')
        with gr.Row():
            with gr.Column():
                person_image = gr.Image(type="pil", label="Person Image", height=800)
                run_button = gr.Button(value="Submit", variant='primary')

            with gr.Column():
                object_image = gr.Image(type="pil", label="Object Image", height=800)
                object_class = gr.Dropdown(label='Object Class', choices=list(args.object_map.keys()))
                enhance_mode = gr.Radio(label='Mode', choices=['Enhanced', 'Baseline'], value='Enhanced')
                optional_prompt = gr.Textbox(label='Optional Prompt', lines=2)

            with gr.Column():
                image_out = gr.Image(type="pil", label="Output", height=800)
                diagnostics = gr.Markdown()

        with gr.Accordion("Advanced ⚙️", open=False):
            guidance_scale = gr.Slider(label="Guidance scale", minimum=1, maximum=50, value=30, step=0.1)
            steps = gr.Slider(label="Steps", minimum=1, maximum=50, value=20, step=1)
            seed = gr.Number(label="Seed", value=-1, precision=0)
            candidate_count = gr.Slider(label="Candidates", minimum=1, maximum=MAX_CANDIDATES, value=1, step=1)

        candidates = gr.Gallery(label="Candidates", columns=2, height=360)

        with gr.Row():
            gr.Examples(
                examples=[
                    [
                        './demo_example/person_top_cloth.jpg',
                        './demo_example/object_top_cloth.jpg', 
                        'top clothes',
                        '',
                    ],
                    [
                        './demo_example/person_bottom_cloth.jpg',
                        './demo_example/object_bottom_cloth.jpg', 
                        'bottom clothes',
                        '',
                    ],
                    [
                        './demo_example/person_dress.jpg',
                        './demo_example/object_dress.jpg', 
                        'dress',
                        '',
                    ],
                    [
                        './demo_example/person_shoes.jpg',
                        './demo_example/object_shoes.jpg', 
                        'shoe',
                        '',
                    ],
                    [
                        './demo_example/person_earrings.jpg',
                        './demo_example/object_earrings.jpg', 
                        'earrings',
                        '',
                    ],
                    [
                        './demo_example/person_bracelet.jpg',
                        './demo_example/object_bracelet.jpg', 
                        'bracelet',
                        '',
                    ],
                    [
                        './demo_example/person_necklace.jpg',
                        './demo_example/object_necklace.jpg', 
                        'necklace',
                        '',
                    ],
                    [
                        './demo_example/person_ring.jpg',
                        './demo_example/object_ring.jpg', 
                        'ring',
                        '',
                    ],
                    [
                        './demo_example/person_sunglasses.jpg',
                        './demo_example/object_sunglasses.jpg', 
                        'sunglasses',
                        '',
                    ],
                    [
                        './demo_example/person_glasses.jpg',
                        './demo_example/object_glasses.jpg', 
                        'glasses',
                        '',
                    ],
                    [
                        './demo_example/person_belt.jpg',
                        './demo_example/object_belt.jpg', 
                        'belt',
                        '',
                    ],
                    [
                        './demo_example/person_bag.jpg',
                        './demo_example/object_bag.jpg', 
                        'bag',
                        '',
                    ],
                    [
                        './demo_example/person_hat.jpg',
                        './demo_example/object_hat.jpg', 
                        'hat',
                        '',
                    ],
                    [
                        './demo_example/person_tie.jpg',
                        './demo_example/object_tie.jpg', 
                        'tie',
                        '',
                    ],
                    [
                        './demo_example/person_bowtie.jpg',
                        './demo_example/object_bowtie.jpg', 
                        'bow tie',
                        '',
                    ],
                ],

                inputs=[person_image, object_image, object_class, optional_prompt],
                examples_per_page=100
            )

        run_button.click(
            generate,
            inputs=[
                person_image,
                object_image,
                object_class,
                optional_prompt,
                steps,
                guidance_scale,
                seed,
                candidate_count,
                enhance_mode,
            ],
            outputs=[image_out, candidates, diagnostics],
        )
    
    demo.launch(
        server_name=os.environ.get("GRADIO_SERVER_NAME", "0.0.0.0"),
        server_port=_env_int("GRADIO_SERVER_PORT", 7860),
        share=os.environ.get("GRADIO_SHARE", "0").lower() in {"1", "true", "yes"},
    )
