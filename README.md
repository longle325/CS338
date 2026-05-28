# OmniTry++ CS338

Mask-free virtual try-on demo and benchmark code based on
**OmniTry: Virtual Try-On Anything without Masks**. The repository keeps the
original FLUX.1-Fill + OmniTry LoRA inference path and adds a lightweight
inference-time improvement layer:

**Geo-Affordance Candidate Selection (GACS)**.

GACS is the method this project should present first. It does not require new
model weights. It improves the pretrained OmniTry pipeline by adding
category-specific geometry prompts, generating multiple candidates, and
selecting the best candidate with reproducible local diagnostics.

## TL;DR

- Use `Enhanced` mode, also shown as `Pretrained + Geometry`, for the improved
  method.
- Use `Baseline` or `Pretrained` only as the original OmniTry comparison.
- GACS is training-free and works with the frozen OmniTry LoRA checkpoint.
- The strongest reproducible comparison is the 32-item hard small-object
  benchmark: GACS improves the proxy total score from `0.623209` to `0.623760`,
  with 17 wins, 14 ties, and 1 loss.
- On the 360-case OmniTry-Bench protocol, the current geometry run completes all
  360 items and reaches official-style metrics close to the paper OmniTry row.
- Fine-tuning is included as an exploratory branch, but it is not the main
  claim yet. The current raw fine-tuned LoRA pilot scores below the pretrained
  baseline on the available benchmark.

## Method

GACS targets failure modes that are common for small or geometry-sensitive
objects: rings, bracelets, earrings, necklaces, glasses, hats, belts, bags, and
shoes. These objects often fail because the model preserves the person but
places the object at the wrong scale or location.

The improved path adds four steps on top of pretrained OmniTry:

1. Build a geometry-aware prompt from the object class.
2. Generate `K` candidates with the same frozen model.
3. Score each candidate with object consistency, person preservation, and
   artifact health.
4. Return the best candidate and save diagnostics.

The current proxy score is:

```text
0.35 * object_consistency + 0.35 * person_preservation + 0.30 * artifact_health
```

This score is intentionally simple. It is useful for reproducible selection and
ablation, but it is not a substitute for human preference evaluation.

## When To Use Each Mode

| Mode | What it does | Recommended use |
|---|---|---|
| `Baseline` / `Pretrained` | Original OmniTry class prompt, one candidate | Control run and speed checks |
| `Enhanced` / `Pretrained + Geometry` | Geometry-aware prompt, multiple candidates, score-and-select | Main method and demo default |
| Fine-tuned LoRA pilot | Experimental LoRA branch trained on pseudo-pairs | Negative/extension result, not main claim |

For demos, start with `Enhanced` and `K=2` or `K=3`. Use `K=1` if GPU memory or
runtime is tight.

## Results

### Hard Small-Object Benchmark

This is the cleanest controlled comparison because the baseline and GACS use the
same pretrained checkpoint. Scores come from `scripts/run_tryon_benchmark.py`
and are proxy selection scores, not official paper metrics.

| Run | Items | Total | Object | Person | Artifact |
|---|---:|---:|---:|---:|---:|
| Pretrained, K=1 | 32 | 0.623209 | 0.255470 | 0.976477 | 0.640091 |
| Pretrained + GACS, K=2 | 32 | 0.623760 | 0.255866 | 0.977028 | 0.640823 |
| Delta | 32 | +0.000551 | +0.000396 | +0.000551 | +0.000732 |
| Raw fine-tuned LoRA pilot | 32 | 0.621025 | 0.252481 | 0.975014 | 0.638004 |

Summary: GACS is small but stable on this benchmark. It wins 17 cases, ties 14,
and loses 1. The fine-tuned pilot is useful evidence that the project should be
framed around geometry-first inference rather than claiming a better trained
model.

### 360-Case OmniTry-Bench Run

The current full run is:

```text
outputs/tryon_benchmark/paper360_pretrained_geo_geo2_20260527_054323
```

Internal proxy summary:

| Run | Items | Total | Object | Person | Artifact |
|---|---:|---:|---:|---:|---:|
| Pretrained + Geometry, K=2 | 360 | 0.600807 | 0.261986 | 0.926764 | 0.615815 |

Official-style metrics produced by `scripts/run_official_omnitry_metrics.py`:

| Method | Items | M-DINO | M-CLIP-I | LPIPS | SSIM | G-Acc. | CLIP-T |
|---|---:|---:|---:|---:|---:|---:|---:|
| OmniTry paper reference | - | 0.6160 | 0.8327 | 0.0542 | 0.9333 | 0.9972 | 0.2831 |
| Pretrained + Geometry | 360 | 0.6050 | 0.8306 | 0.0740 | 0.9195 | 1.0000 | 0.2794 |

Interpretation: the 360-case run is complete and useful for reporting, but
avoid claiming a definitive paper-level win. The local official metric pipeline,
mask generation, hardware, seeds, and implementation details can still differ
from the original paper setup.

### Example Cases Where Geometry Helps

These cases are good demo candidates because GACS improves placement or
selection while preserving the person image.

| Case | Class | Set | Pretrained | Pretrained + Geo | Delta |
|---|---|---|---:|---:|---:|
| `ring_woman_015_204` | ring | hard benchmark win | 0.641162 | 0.644585 | +0.003423 |
| `ring_woman_011_203` | ring | diverse demo | 0.618754 | 0.621918 | +0.003164 |
| `bracelet_woman_008_302` | bracelet | hard benchmark win | 0.670664 | 0.673440 | +0.002776 |
| `ring_woman_015_102` | ring | hard benchmark win | 0.524317 | 0.526425 | +0.002108 |
| `earrings_woman_004_103` | earrings | diverse demo | 0.534739 | 0.536624 | +0.001885 |
| `glasses_woman_010_301` | glasses | diverse demo | 0.598226 | 0.600092 | +0.001866 |
| `necklace_woman_012_101` | necklace | diverse demo | 0.526438 | 0.528071 | +0.001633 |

Generated demo artifacts:

- `outputs/demo/geo_method_report/index.html`
- `outputs/demo/geo_method_report/geo_method_demo.mp4`
- `outputs/demo/geo_method_report/demo_manifest.json`

## Quick Start

### Hardware

Recommended for the default demo:

- 1 GPU with at least 40 GB VRAM, preferably 48 GB.
- Good cloud choices: L40S 48 GB, A40 48 GB, A100 40 GB/80 GB.
- Host RAM: 64 GB minimum, 128 GB preferred.
- Disk: 150-250 GB because FLUX checkpoints are large.
- CUDA 12.x image.

The upstream OmniTry README reports about 28 GB VRAM for bfloat16 inference. A
24 GB GPU may fail unless you lower resolution or use CPU offload aggressively.

### Setup

Clone the repository:

```bash
git clone https://github.com/longle325/CS338.git
cd CS338
```

FLUX.1-Fill-dev is a gated Hugging Face model. Accept the model license on
Hugging Face, create a token, then export it:

```bash
export HF_TOKEN=hf_your_token_here
```

Recommended environment name for this repo:

```bash
OMNITRY_ENV_NAME=cs338 bash scripts/setup_omnitry.sh
export OMNITRY_ENV_NAME=cs338
```

The setup script creates or updates the conda environment, installs Python
dependencies, downloads `black-forest-labs/FLUX.1-Fill-dev`, downloads
`Kunbyte/OmniTry`, and checks GPU/checkpoint status.

If you prefer the default upstream-style env name, run:

```bash
bash scripts/setup_omnitry.sh
```

Some benchmark helper scripts currently activate `cs338`, so `cs338` is the
least surprising environment name for reproducing the numbers in this repo.

### Run The Gradio Demo

```bash
bash scripts/run_gradio.sh
```

Default URL:

```text
http://SERVER_IP:7860
```

Optional launch overrides:

```bash
GRADIO_SERVER_PORT=8080 bash scripts/run_gradio.sh
GRADIO_SHARE=1 bash scripts/run_gradio.sh
```

In the UI, select:

- `Enhanced` for the geometry method.
- `Baseline` for the original OmniTry comparison.

### Run The FastAPI Backend

```bash
bash demo/backend/run.sh
```

Default URL:

```text
http://localhost:8010
```

Example API call:

```bash
curl -X POST http://localhost:8010/api/v1/runs/compare \
  -F person_image=@demo_example/person_ring.jpg \
  -F object_image=@demo_example/object_ring.jpg \
  -F object_class=ring \
  -F geometry_candidate_count=2
```

See `demo/backend/README.md` for the full API.

### Run The GPU-Free Cached Video Demo

Use this mode when recording a presentation on a laptop or classroom machine
without GPU. It serves the React UI only, waits about 10 seconds, and then
returns cached left/right outputs so the flow still looks like a real inference
request.

```bash
bash scripts/run_cached_video_demo.sh
```

Upload matching files from:

```text
demo/frontend/public/demo-cache-inputs
```

Recommended recording cases:

| Case | Person input | Item input |
|---|---|---|
| `01_ring_strong` | `01_ring_strong_person.jpg` | `01_ring_strong_object.jpg` |
| `02_bracelet_strong` | `02_bracelet_strong_person.jpg` | `02_bracelet_strong_object.jpg` |
| `04_glasses` | `04_glasses_person.jpg` | `04_glasses_object.jpg` |
| `05_earrings` | `05_earrings_person.jpg` | `05_earrings_object.jpg` |

The UI result tab shows only two panels: left is pretrained OmniTry, right is
the geometry-selected output. This keeps the recording focused on the method
comparison.

## Reproduce Benchmarks

Activate the environment:

```bash
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate cs338
```

Download OmniTry-Bench metadata:

```bash
python scripts/download_omnitry_data.py
```

Download full OmniTry-Bench images:

```bash
python scripts/download_omnitry_data.py --full
```

Build a hard-case manifest:

```bash
python scripts/build_hard_cases.py \
  --bench-root data/OmniTry_Bench \
  --top-k 300 \
  --per-class 40 \
  --output data/hard_cases/omnitry_hard_cases.json
```

Run a 32-item baseline:

```bash
CUDA_VISIBLE_DEVICES=0 python scripts/run_tryon_benchmark.py \
  --manifest data/hard_cases/omnitry_full_local_hard_cases.json \
  --output-dir outputs/tryon_benchmark/original_enhanced \
  --summary-output outputs/tryon_benchmark/original_enhanced_summary.json \
  --mode Baseline \
  --candidate-count 1 \
  --max-items 32
```

Run a 32-item GACS benchmark:

```bash
CUDA_VISIBLE_DEVICES=0 python scripts/run_tryon_benchmark.py \
  --manifest data/hard_cases/omnitry_full_local_hard_cases.json \
  --output-dir outputs/tryon_benchmark/original_enhanced_c2 \
  --summary-output outputs/tryon_benchmark/original_enhanced_c2_summary.json \
  --mode Enhanced \
  --candidate-count 2 \
  --max-items 32
```

Run the paper-style 360-case benchmark across multiple GPUs:

```bash
PAPER360_GPUS=0,1,2,3 MIN_FREE_MB=28000 bash scripts/run_paper360_benchmark.sh
```

Run official-style mask and metric evaluation for a completed result directory:

```bash
RESULT_DIR=outputs/tryon_benchmark/paper360_pretrained_geo_geo2_20260527_054323 \
  bash scripts/run_official_omnitry_pipeline.sh
```

Regenerate the project report and demo case page:

```bash
python scripts/build_geo_method_report.py
```

Detailed training and extension plan:

- `docs/enhance_training_plan.md`
- `docs/geo_affordance_project_report.md`

## Fine-Tuning Status

The repository contains a pilot FLUX LoRA fine-tuning path:

```bash
python scripts/build_paired_manifest.py \
  --index data/OmniTry_Bench/omni_vtryon_bench_v1.json \
  --top-k 300 \
  --per-class 40 \
  --output data/hard_cases/omnitry_pseudo_paired_train.json

CUDA_VISIBLE_DEVICES=0 accelerate launch --num_processes 1 scripts/train_geo_lora.py \
  --manifest data/hard_cases/omnitry_pseudo_paired_train.json \
  --output checkpoints/enhance/omnitry_geo_lora.safetensors \
  --resolution 512 \
  --max-train-steps 1000 \
  --train-batch-size 1 \
  --gradient-accumulation-steps 4
```

Each paired item must include `person_path`, `object_path`, and `target_path`
or `gt_path`. OmniTry-Bench is primarily an evaluation benchmark and does not
provide true supervised target pairs for every case. The current pseudo-pair
path is useful for experimentation, but the measured result is not strong enough
to present fine-tuning as the main improvement.

## Limitations

- Gains are modest because GACS does not change the generator; it selects better
  samples from the same model.
- The local proxy score can prefer clean images that still miss subtle semantic
  details. Human evaluation is still needed.
- Small reflective or thin objects remain difficult: rings, earrings, chains,
  bracelet links, and glasses arms can be missed or distorted.
- Placement can fail when the expected body landmark is occluded, cropped, or
  outside the frame.
- Official metrics depend on generated masks. Mask errors can affect M-DINO,
  M-CLIP-I, LPIPS, SSIM, and G-Accuracy.
- Fine-tuning is blocked by data quality. Crawled images rarely provide the
  required person/object/target triplets with clean masks.
- Full FLUX inference is expensive. Multi-candidate generation improves
  reliability but increases runtime roughly linearly with `K`.

## Project Layout

```text
configs/                  Model and checkpoint config
demo/backend/             FastAPI live comparison backend
demo/frontend/            Optional Vite/React demo frontend
demo_example/             Built-in demo person/object pairs
docs/                     Project plan and report
omnitry/enhance/          GACS prompts, scoring, data helpers, training stubs
outputs/tryon_benchmark/  Saved benchmark outputs and summaries
scripts/                  Setup, benchmark, official metrics, and training tools
```

## Low VRAM Knobs

Lower the working resolution before launch:

```bash
OMNITRY_MAX_AREA=589824 bash scripts/run_gradio.sh  # 768 * 768
OMNITRY_MAX_AREA=262144 bash scripts/run_gradio.sh  # 512 * 512
```

Keep `K=1` when VRAM or runtime is tight. Candidate generation is sequential, so
larger `K` mainly increases time, but hard cases can still become expensive.

## Manual Checkpoint Layout

If you download checkpoints manually, keep this layout:

```text
checkpoints/
  FLUX.1-Fill-dev/
    transformer/
    text_encoder/
    text_encoder_2/
    tokenizer/
    tokenizer_2/
    vae/
  omnitry_v1_unified.safetensors
```

The paths are configured in:

```text
configs/omnitry_v1_unified.yaml
```

## Troubleshooting

### Hugging Face download fails

Check that `HF_TOKEN` is exported, the FLUX.1-Fill-dev license has been accepted
on Hugging Face, and the machine has enough disk/network access.

### CUDA out of memory

Use a 40-48 GB GPU, lower `OMNITRY_MAX_AREA`, keep candidates at `K=1`, enable
CPU offload where available, and close other GPU processes.

### Gradio opens locally but not in browser

On Vast.ai or a remote GPU server, expose or tunnel port `7860`. You can also
set:

```bash
GRADIO_SHARE=1 bash scripts/run_gradio.sh
```

## References

- Paper: https://arxiv.org/abs/2508.13632
- Upstream repo: https://github.com/Kunbyte-AI/OmniTry
- FLUX.1-Fill-dev: https://huggingface.co/black-forest-labs/FLUX.1-Fill-dev
- OmniTry weights: https://huggingface.co/Kunbyte/OmniTry
