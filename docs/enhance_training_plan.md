# OmniTry++ Enhance Training Plan

## Implemented Now

The current repo implements the runnable pieces needed before full FLUX fine-tuning:

1. OmniTry-Bench downloader: `scripts/download_omnitry_data.py`
2. hard-case manifest builder: `scripts/build_hard_cases.py`
3. licensed raw candidate crawler: `scripts/crawl_hard_cases.py`
4. lightweight Affordance Planner model: `omnitry/enhance/planner.py`
5. planner trainer: `scripts/train_affordance_planner.py`
6. planner evaluator: `scripts/eval_affordance_planner.py`
7. end-to-end smoke test: `scripts/run_enhance_smoke.sh`

This is not yet full GeoTrace-ID OmniTry++ training. It is Stage 0/1 infrastructure: data selection, weak spatial labels, planner train/eval, and demo integration.

## Stages

### Stage 0: Data Acquisition

Inputs:

- OmniTry-Bench metadata/full images from Hugging Face.
- Optional Wikimedia Commons hard-case raw candidates with license metadata.
- Demo examples for local smoke tests.

Outputs:

- `data/OmniTry_Bench/`
- `data/hard_cases/omnitry_hard_cases.json`
- `data/hard_cases/commons_hard_cases.json`

Smoke command:

```bash
python scripts/download_omnitry_data.py
python scripts/build_hard_cases.py --demo-fallback --require-local-images --top-k 8
python scripts/crawl_hard_cases.py --max-per-query 1
```

Full command:

```bash
python scripts/download_omnitry_data.py --full
python scripts/build_hard_cases.py --bench-root data/OmniTry_Bench --top-k 300 --per-class 40
```

### Stage 1: Affordance Planner

Goal:

Predict a soft placement heatmap from person image + object class. The current target is a weak taxonomy-derived heatmap. In the stronger version, replace this target with pseudo masks/boxes from GroundingDINO/SAM plus pose/hand landmarks.

Outputs:

- `checkpoints/enhance/affordance_planner.pt`
- `outputs/enhance/planner_train_metrics.json`
- `outputs/enhance/planner_eval.json`

Smoke command:

```bash
bash scripts/run_enhance_smoke.sh
```

Full command:

```bash
python scripts/train_affordance_planner.py \
  --manifest data/hard_cases/omnitry_hard_cases.json \
  --output checkpoints/enhance/affordance_planner.pt \
  --epochs 20 \
  --batch-size 16

python scripts/eval_affordance_planner.py \
  --manifest data/hard_cases/omnitry_hard_cases.json \
  --checkpoint checkpoints/enhance/affordance_planner.pt
```

Expected time:

- smoke CPU: less than 1 minute
- 300-1,000 local images: about 10-60 minutes on one 48 GB GPU
- pseudo-label generation with GroundingDINO/SAM: usually hours, depending on image count

### Stage 2: FLUX LoRA/GeometryAdapter Fine-Tuning

Status: not implemented yet.

Goal:

Inject planner heatmaps/geometry priors into the diffusion model and fine-tune LoRA/adapters on hard cases.

Needed next files:

- `omnitry/enhance/conditioning.py`
- `scripts/train_geo_lora.py`
- `scripts/run_tryon_benchmark.py`

Expected time:

- small LoRA pilot: 12-36 hours on one 48-80 GB GPU
- medium 10k-25k step run: 2-5 days on one A100/L40S class GPU
- research-grade run: multi-GPU recommended

## Benchmark

Use three levels:

1. OmniTry-Bench small: 360 pairs for fast regression.
2. OmniTry-Bench full: 6,975 pairs for more stable numbers.
3. Hard-case subset: jewelry, watches, glasses, hats, bags, shoes, side/back/profile/hand-occlusion cases.

Compare:

- Baseline: original prompt, one seed.
- Enhanced inference: affordance prompt + generate-and-rerank.
- Enhanced trained planner: planner-guided prompt/conditioning once Stage 2 exists.

Metrics:

- Object consistency: M-DINO, M-CLIP-I.
- Person preservation: LPIPS, SSIM outside object region.
- Localization: G-Accuracy, CLIP-T.
- Hard-case detail: crop-DINO, color histogram similarity, edge consistency.
- Human A/B preference on 100-300 hard cases.

Minimum success bar:

- +3-5% object consistency on hard classes.
- +5% localization success on hard classes.
- no meaningful degradation in person preservation.
- >55-60% human preference versus baseline.
