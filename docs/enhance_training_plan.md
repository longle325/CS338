# OmniTry++ Enhance Training Plan

## Current Project Recommendation

Use the geometry method as the main project contribution. The polished report is now in
`docs/geo_affordance_project_report.md`, with demo assets in
`outputs/demo/geo_method_report/`.

The strongest completed result is pretrained OmniTry versus pretrained OmniTry plus
Geo-Affordance Candidate Selection: 0.623209 -> 0.623760 on the 32-item hard
small-object benchmark, with 17 wins, 14 ties, and 1 loss. The fine-tuning branch
should be presented as an exploratory negative result because the current raw
fine-tuned LoRA scores lower than the pretrained baseline on the same benchmark.

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

### Stage 2: FLUX LoRA Fine-Tuning

Status: pilot trainer implemented in `scripts/train_geo_lora.py`.

Goal:

Continue fine-tuning the OmniTry dual-stream FLUX LoRA adapters on paired try-on data. The trainer preserves the existing
two-stream inference design: one internal stream uses `vtryon_lora` and the other uses `garment_lora`.

Important data requirement:

- OmniTry-Bench is an evaluation benchmark. It has `person_path`, `object_path`, and target captions, but it does not
  include target try-on images.
- Real supervised LoRA fine-tuning therefore needs a manifest where each item also has `target_path` or `gt_path`.
- `--allow-person-target` exists only for reconstruction smoke tests and should not be reported as try-on fine-tuning.

Minimal paired manifest item:

```json
{
  "id": "example_0001",
  "category": "bag",
  "person_path": "path/to/person_without_bag.jpg",
  "object_path": "path/to/bag_reference.jpg",
  "target_path": "path/to/person_wearing_bag.jpg"
}
```

Validate paired data:

```bash
python scripts/train_geo_lora.py \
  --manifest data/hard_cases/paired_tryon_train.json \
  --validate-data-only
```

Build a pseudo-paired pilot manifest from OmniTry-Bench person masks:

```bash
python scripts/build_paired_manifest.py \
  --index data/OmniTry_Bench/omni_vtryon_bench_v1.json \
  --top-k 300 \
  --per-class 40 \
  --output data/hard_cases/omnitry_pseudo_paired_train.json
```

This creates:

- source person: original person image with the masked object region erased,
- object reference: crop from the original masked region,
- target: original person image.

This is useful for a LoRA reconstruction pilot, but it is not a substitute for true object-transfer paired data.

Launch a small multi-GPU LoRA pilot:

```bash
CUDA_VISIBLE_DEVICES=2,3,7 accelerate launch --num_processes 3 scripts/train_geo_lora.py \
  --manifest data/hard_cases/omnitry_pseudo_paired_train.json \
  --output checkpoints/enhance/omnitry_geo_lora.safetensors \
  --metrics-output outputs/enhance/geo_lora_train_metrics.json \
  --resolution 512 \
  --max-train-steps 1000 \
  --train-batch-size 1 \
  --gradient-accumulation-steps 4
```

Smoke-test the trainer's data path only with current OmniTry-Bench files:

```bash
python scripts/train_geo_lora.py \
  --manifest data/hard_cases/omnitry_full_local_hard_cases.json \
  --validate-data-only \
  --allow-person-target \
  --max-items 4
```

Run a lightweight generation benchmark:

```bash
CUDA_VISIBLE_DEVICES=2 python scripts/run_tryon_benchmark.py \
  --manifest data/hard_cases/omnitry_full_local_hard_cases.json \
  --output-dir outputs/tryon_benchmark/enhanced \
  --summary-output outputs/tryon_benchmark/enhanced_summary.json \
  --mode Enhanced \
  --max-items 32 \
  --candidate-count 1
```

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

## Current Pilot Result

As of the latest 32-item hard-case pilot:

- Pretrained OmniTry LoRA: total mean `0.623209`
- Geo LoRA checkpoint at step 850: total mean `0.621025`
- Geometry candidate/rerank best-of summary: total mean `0.623845`
- Rerank delta versus pretrained: `+0.000636`
- Best-of source mix: 14 original candidate-2 outputs, 5 fine-tuned outputs, 13 original outputs

The raw step-850 fine-tuned checkpoint does not beat the pretrained checkpoint on the full mean yet. The useful result is the
geometry-aware selection layer: it can recover a small positive benchmark gain while preserving cases where the pretrained model
is still stronger.

## Stage 3: Geo-Affordance Candidate Selection

Working name: Geo-Affordance Candidate Selection (GACS).

Theory:

The enhanced pseudo-paired dataset turns each example into a localized reconstruction problem: erase the object affordance region,
provide the object crop as reference, and train the adapter to restore the original target. This biases the adapter toward
placement geometry instead of broad image repainting. At inference time, generate multiple candidates and select the one that
maximizes a geometry consistency score:

```text
score = 0.35 * object_consistency
      + 0.35 * person_preservation
      + 0.30 * artifact_health
```

- `object_consistency`: color/detail histogram match between the object reference and the predicted affordance crop.
- `person_preservation`: low pixel change outside the affordance region.
- `artifact_health`: sharpness, contrast, and pixel-range sanity checks.

This is a defensible inference-time trick because it does not invent a new target label; it reuses the same affordance assumption
used to build the enhanced pseudo-paired data. For reporting, call the raw trained adapter "Geo LoRA" and the final benchmark
system "Geo LoRA + GACS reranking".

Demo artifacts:

- UI: `outputs/demo/geo_affordance_finetune_wins/index.html`
- Video: `outputs/demo/geo_affordance_finetune_wins/finetune_wins.mp4`
- Fine-tuned winner manifest: `outputs/demo/geo_affordance_finetune_wins/winner_manifest.json`
- Method note: `outputs/demo/geo_affordance_finetune_wins/method.md`

## Three-Way Demo Protocol

Use a diverse-person small-object subset instead of taking the first sequential benchmark rows:

- Manifest: `data/hard_cases/omnitry_small_object_diverse_demo.json`
- Categories: bracelet, ring, earrings, necklace, glasses, sunglasses
- Inference card: `CUDA_VISIBLE_DEVICES=2`
- Demo steps: 8, for fast reproducible visual comparison

Report three rows:

1. `Pretrained`: original OmniTry LoRA, baseline prompt, one candidate.
2. `Pretrained + GACS`: original OmniTry LoRA, geometry-enhanced prompt, two candidates, GACS selection.
3. `GACS + small-object data`: GACS over a candidate pool that includes the original pretrained candidate and the
   step-850 Geo LoRA trained on 300 small/accessory pseudo-pairs.

Important data note:

- The current Geo LoRA was trained on `data/hard_cases/omnitry_pseudo_paired_train.json`, not on crawled Commons images.
- No `commons_hard_cases.json` or downloaded crawl manifest is present in the current workspace.
- Commons/Wikimedia crawl outputs are raw licensed image candidates. They should not be reported as training data until they
  are filtered and converted into masked pseudo-pairs or supervised pairs.

## LLM Labeling Cost Guard

Before using a paid vision LLM to label crawled images, run the labeler through the backend budget guard:

```bash
python scripts/label_crawled_images.py \
  --input data/hard_cases/commons_hard_cases.json \
  --output data/hard_cases/commons_llm_labels.json \
  --dry-run
```

Actual labeling:

```bash
python scripts/label_crawled_images.py \
  --input data/hard_cases/commons_hard_cases.json \
  --output data/hard_cases/commons_llm_labels.json
```

Budget controls:

- `LLM_LABEL_BUDGET_USD=100`: hard cap.
- `LLM_LABEL_SOFT_BUDGET_USD=95`: stop-before-next-request threshold.
- `LLM_LABEL_INPUT_USD_PER_1M` / `LLM_LABEL_OUTPUT_USD_PER_1M`: model price assumptions for cost estimates.
- `LLM_LABEL_IMAGE_INPUT_TOKENS`: image-token equivalent used before the provider returns real usage.
- State: `outputs/llm_labeling/cost_state.json`
- Event log: `outputs/llm_labeling/cost_events.jsonl`

The guard reserves estimated cost before the HTTP request. If the provider returns token usage, the ledger adjusts the reserved
estimate to the actual usage-derived estimate. If the next request would reach the soft or hard cap, the script blocks before
calling the API.

Current Commons labeling run:

- Crawl output: `data/hard_cases/commons_hard_cases.json`
- LLM labels: `data/hard_cases/commons_llm_labels.json`
- Usable-filtered labels: `data/hard_cases/commons_llm_usable_labels.json`
- Items labeled: 38
- Usable after LLM filtering: 10
- Ledger spend estimate after token-usage adjustment: `$0.666060`
- Budget blocks: 0

Three-way demo artifacts:

- UI: `outputs/demo/geo_affordance_three_way/index.html`
- Video: `outputs/demo/geo_affordance_three_way/three_way_demo.mp4`
- Manifest: `outputs/demo/geo_affordance_three_way/demo_manifest.json`

Minimum success bar:

- +3-5% object consistency on hard classes.
- +5% localization success on hard classes.
- no meaningful degradation in person preservation.
- >55-60% human preference versus baseline.
