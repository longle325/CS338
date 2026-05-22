# OmniTry CS338

This repository is a runnable OmniTry demo with a small inference-side improvement layer:

- mask-free virtual try-on for clothes, shoes, jewelry, and accessories,
- optional affordance prompt hints per object class,
- multi-candidate generation with lightweight reranking,
- confidence diagnostics for the selected result,
- setup and launch scripts for GPU machines such as Vast.ai.

The code is based on **OmniTry: Virtual Try-On Anything without Masks** and keeps the original FLUX.1-Fill + OmniTry LoRA inference path.

## Hardware

Recommended for the default demo:

- **1 GPU with at least 40 GB VRAM**, preferably **48 GB**.
- Good choices on Vast.ai: **L40S 48GB**, **A40 48GB**, **A100 40GB/80GB**.
- Host RAM: **64 GB minimum**, **128 GB preferred**.
- Disk: **150-250 GB** because FLUX checkpoints are large.
- CUDA 12.x image.

The original OmniTry README reports about **28 GB VRAM** for bfloat16 inference. A 24 GB GPU may fail unless you lower resolution/offload aggressively.

## One-Time Setup

Clone the repo on the GPU machine:

```bash
git clone https://github.com/longle325/CS338.git
cd CS338
```

FLUX.1-Fill-dev is a gated Hugging Face model. Accept the model license on Hugging Face, create a token, then export it:

```bash
export HF_TOKEN=hf_your_token_here
```

Run setup:

```bash
bash scripts/setup_omnitry.sh
```

The setup script will:

1. create or update the `omnitry` conda environment,
2. install Python dependencies,
3. download `black-forest-labs/FLUX.1-Fill-dev` into `checkpoints/FLUX.1-Fill-dev`,
4. download `Kunbyte/OmniTry` LoRA into `checkpoints/omnitry_v1_unified.safetensors`,
5. print GPU/checkpoint status.

If checkpoints are already present, rerunning setup is safe.

## Run

```bash
bash scripts/run_gradio.sh
```

Default URL:

```text
http://SERVER_IP:7860
```

For Vast.ai, open the exposed HTTP port for `7860` or tunnel it with SSH, depending on the instance template.

Optional launch overrides:

```bash
GRADIO_SERVER_PORT=8080 bash scripts/run_gradio.sh
GRADIO_SHARE=1 bash scripts/run_gradio.sh
```

## Low VRAM Knobs

For GPUs close to the limit, lower the working resolution before launch:

```bash
OMNITRY_MAX_AREA=589824 bash scripts/run_gradio.sh  # 768 * 768
OMNITRY_MAX_AREA=262144 bash scripts/run_gradio.sh  # 512 * 512
```

Keep `Candidates = 1` in the UI when VRAM or runtime is tight. Candidate generation is sequential, so it mainly increases time, but hard cases can become expensive quickly.

## Enhance Training Pipeline

The UI has two modes:

- `Baseline`: the original class prompt and one candidate.
- `Enhanced`: affordance prompt hints, multi-candidate generation, reranking, and confidence diagnostics.

The repository also includes the first training scaffold for the research enhancement path: a lightweight **Affordance Planner**. It learns a placement heatmap from person image + object class. The current target is a weak affordance heatmap generated from the class taxonomy; for a full research run, replace or augment it with GroundingDINO/SAM masks, pose/hand landmarks, and depth pseudo-labels.

Smoke test the whole enhance pipeline without FLUX checkpoints:

```bash
bash scripts/run_enhance_smoke.sh
```

This creates a tiny demo hard-case manifest, trains the planner for one CPU epoch, and evaluates heatmap Dice/IoU. It only proves that the training/eval code runs.

Download OmniTry-Bench metadata:

```bash
conda activate omnitry
python scripts/download_omnitry_data.py
```

Download the full OmniTry-Bench images:

```bash
python scripts/download_omnitry_data.py --full
```

Build a hard-case manifest from OmniTry-Bench:

```bash
python scripts/build_hard_cases.py \
  --bench-root data/OmniTry_Bench \
  --top-k 300 \
  --per-class 40 \
  --output data/hard_cases/omnitry_hard_cases.json
```

Crawl additional licensed raw candidates from Wikimedia Commons:

```bash
python scripts/crawl_hard_cases.py \
  --max-per-query 10 \
  --download \
  --output data/hard_cases/commons_hard_cases.json
```

Train the planner:

```bash
python scripts/train_affordance_planner.py \
  --manifest data/hard_cases/omnitry_hard_cases.json \
  --output checkpoints/enhance/affordance_planner.pt \
  --epochs 20 \
  --batch-size 16
```

Evaluate the planner:

```bash
python scripts/eval_affordance_planner.py \
  --manifest data/hard_cases/omnitry_hard_cases.json \
  --checkpoint checkpoints/enhance/affordance_planner.pt \
  --output outputs/enhance/planner_eval.json
```

Expected training time:

- smoke CPU run: less than a minute,
- planner on 300-1,000 items: roughly 10-60 minutes on a 48 GB GPU,
- planner with real pseudo-label generation: hours, depending on GroundingDINO/SAM/pose/depth throughput,
- FLUX LoRA/GeometryAdapter training is not included yet and is the next phase after planner + benchmark harness.

How to know it is better:

- planner: compare heatmap Dice/IoU against pseudo masks or held-out weak labels,
- try-on output: run Baseline vs Enhanced on OmniTry-Bench small/full with the same seeds, steps, and guidance,
- report M-DINO/M-CLIP-I for object consistency, LPIPS/SSIM for person preservation, G-Accuracy/CLIP-T for localization, and a hard-case human preference pass for jewelry, bags, shoes, hats, and glasses.

Detailed stage plan: [docs/enhance_training_plan.md](docs/enhance_training_plan.md)

## Check Runtime

```bash
conda activate omnitry
python scripts/check_runtime.py
```

This prints detected GPUs, VRAM, and checkpoint availability.

## Manual Checkpoint Layout

If you download checkpoints yourself, keep this layout:

```text
checkpoints/
├── FLUX.1-Fill-dev/
│   ├── transformer/
│   ├── text_encoder/
│   ├── text_encoder_2/
│   ├── tokenizer/
│   ├── tokenizer_2/
│   ├── vae/
│   └── ...
└── omnitry_v1_unified.safetensors
```

The paths are configured in:

```text
configs/omnitry_v1_unified.yaml
```

## What Changed From Upstream

- The Gradio demo lazy-loads the model only when generation starts.
- Missing checkpoints now produce a clear error instead of a cryptic model-load failure.
- Inputs are normalized to RGB, including transparent object images.
- Object-class affordance prompts improve placement/scale/occlusion instructions.
- Optional user prompt is appended without replacing the class prompt.
- Multiple candidates can be generated and reranked with local image-stat heuristics.
- The UI shows candidate gallery, best seed, score table, and confidence label.

## Troubleshooting

### Hugging Face download fails

Make sure:

- `HF_TOKEN` is exported,
- your Hugging Face account accepted the FLUX.1-Fill-dev license,
- the instance has enough disk and network access.

### CUDA out of memory

Use a 40-48 GB GPU, lower `OMNITRY_MAX_AREA`, keep candidates at 1, and close other GPU processes.

### Gradio opens locally but not in browser

On Vast.ai, expose or tunnel port `7860`. You can also set:

```bash
GRADIO_SHARE=1 bash scripts/run_gradio.sh
```

## References

- Paper: https://arxiv.org/abs/2508.13632
- Upstream repo: https://github.com/Kunbyte-AI/OmniTry
- FLUX.1-Fill-dev: https://huggingface.co/black-forest-labs/FLUX.1-Fill-dev
- OmniTry weights: https://huggingface.co/Kunbyte/OmniTry
