# OmniTry++ PRD

## Context

This PRD is derived from `docs/OmniTry_diem_yeu_va_de_xuat_cai_tien.docx` and a quick check against the public OmniTry paper/repository. The original OmniTry design is strong because it keeps the user-facing flow mask-free while using FLUX.1-Fill, two LoRA streams, masked/full attention routing, and paired object references. The main weaknesses are data-prior dependence, long-tail imbalance, erased-image shortcuts, weak explicit geometry, small-object identity loss, narrow evaluation, and expensive inference.

The current repository is inference/demo-only. It does not include the training data engine, trainer, or metric pipeline needed to build the full GeoTrace-ID OmniTry++ research system in one pass. The first implementation should therefore improve the deployable inference surface while leaving clear extension points for future planner/geometry/identity modules.

## Goals

1. Preserve the current mask-free user flow: person image + object image + object class, with optional prompt.
2. Add an inference-side affordance layer that turns object class into stronger placement/occlusion/person-preservation prompt constraints.
3. Add generate-and-rerank support for hard cases by sampling multiple seeds and selecting the best candidate with lightweight, dependency-free QA heuristics.
4. Expose confidence and candidate metadata so weak outputs are visible instead of silently trusted.
5. Make the demo safer for practical use with input validation, RGB normalization, deterministic seed handling, and clear advanced controls.

## Non-Goals

1. Do not claim to implement trained Affordance-Geometry Planner, GeometryAdapter, Trace-Adversarial Eraser, or Multi-scale Object Identity Encoder in this pass.
2. Do not add heavy new model dependencies such as GroundingDINO, SAM, CLIP vision, or pose/depth estimators to the Gradio demo.
3. Do not change model weights or LoRA architecture.

## MVP Requirements

### R1. Robust Inference Inputs

- Validate missing person/object images and unknown object class.
- Convert inputs to RGB before tensor transforms.
- Clamp generation parameters to safe ranges.
- Return the actual seed used for reproducibility.

### R2. Affordance Prompting

- Maintain a category-to-affordance table for the supported classes.
- Compose the base object prompt with placement, scale, occlusion, and identity-preservation hints.
- Allow an optional user prompt that augments, not replaces, the category prompt.

### R3. Candidate Generation

- Add a `candidate_count` advanced control.
- Generate K candidates using deterministic seed offsets.
- Keep K bounded to avoid accidental high VRAM/time usage.
- Return a gallery of all candidates with seed/score labels.

### R4. Lightweight QA Reranking

- Score candidates with fast local heuristics:
  - object-color compatibility between object image and generated output,
  - person-preservation proxy against the original person image,
  - blur/contrast/artifact proxy,
  - saturation/extreme-pixel penalty.
- Select the highest-scoring candidate as the primary output.
- Return confidence as `high`, `medium`, or `low` with a compact diagnostics table.

### R5. Demo UX

- Add optional prompt, candidate count, and confidence/diagnostic output.
- Keep existing examples working.
- Keep default behavior equivalent to the old one-candidate flow.

## Future Research Requirements

1. Data Quality & Balance Engine: quality-scored pseudo labels, class-balanced sampler, long-tail memory bank.
2. Affordance-Geometry Planner: heatmaps, anchors, occlusion maps, and deformation priors from pose/hand/depth/caption pseudo labels.
3. Trace-Adversarial Erasing: multi-eraser variants, trace discriminator, and consistency losses.
4. Multi-scale Object Identity Bank: global/local/edge/color object tokens and crop-level refinement.
5. Evaluation Expansion: long-tail set, geometry stress test, small-object ID benchmark, and human/product-detail preference checks.

## Acceptance Criteria

1. `gradio_demo.py` supports the old examples and a new optional prompt without breaking the existing model loading path.
2. `generate(...)` can return one best image, a candidate gallery, and textual diagnostics.
3. The implementation imports successfully in the `omnitry` conda environment.
4. Helper logic is testable without loading the FLUX checkpoint.

## Implementation Plan

1. Refactor inference helpers in `gradio_demo.py` while keeping global model initialization compatible with the original demo.
2. Add category affordance metadata and prompt composition.
3. Add candidate loop and seed schedule.
4. Add dependency-free scoring utilities based on PIL/NumPy/Torch-free image statistics.
5. Update Gradio UI outputs and advanced controls.
6. Run syntax/import-light checks in `conda activate omnitry`.
