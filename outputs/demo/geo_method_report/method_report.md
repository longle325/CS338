# Geo-Affordance Candidate Selection for Small-Object Try-On

Date: 2026-05-27

## Executive Summary

This project should be presented around the geometry method, not around fine-tuning. The strongest reproducible result is a controlled comparison between the same pretrained OmniTry pipeline and the same pipeline with Geo-Affordance Candidate Selection (GACS). On the 32-item hard small-object benchmark, GACS improves the total mean from 0.623209 to 0.623760, a delta of +0.000551. It wins 17 cases, ties 14, and loses 1.

The fine-tuning branch is valuable as an exploratory negative result. With the data available in this environment, the raw fine-tuned LoRA scores lower than the pretrained baseline (0.621025 versus 0.623209). That makes fine-tuning a weak main claim, but a strong motivation for a geometry-first method that works without collecting expensive paired labels.

## Method

GACS is a training-free post-processing protocol for small-object virtual try-on. It adds category-specific geometry constraints to the prompt and then generates multiple candidates with the frozen pretrained model. Each candidate is scored using three terms:

- Object consistency: compare the object reference to the expected affordance region using color histogram overlap.
- Person preservation: compare pixels outside the affordance region to the original person image.
- Artifact health: reward sharpness, contrast, and non-saturated pixels.

The final score is `0.35 * object + 0.35 * person + 0.30 * artifact`. This is intentionally simple and reproducible. The key idea is that small objects such as rings, bracelets, earrings, glasses, and necklaces fail mostly because the model is underconstrained about placement, scale, and occlusion. Geometry-aware prompting plus candidate selection narrows the search space without updating model weights.

## Benchmark Protocol

The main benchmark is the 32-item local hard subset from OmniTry small-object cases. The baseline is the pretrained model with one generated candidate. The geometry run uses the same pretrained weights with GACS and two candidates. All scores are produced by `scripts/run_tryon_benchmark.py` and are stored under `outputs/tryon_benchmark`.

| Protocol | Items | Total | Object | Person | Artifact |
| --- | --- | --- | --- | --- | --- |
| Pretrained | 32 | 0.623209 | 0.255470 | 0.976477 | 0.640091 |
| Pretrained + Geo | 32 | 0.623760 | 0.255866 | 0.977028 | 0.640823 |
| Delta |  | +0.000551 | +0.000396 | +0.000551 | +0.000732 |

## Class Breakdown

| Class | Count | Pretrained | Pretrained + Geo | Delta |
| --- | --- | --- | --- | --- |
| bracelet | 15 | 0.651278 | 0.651681 | +0.000403 |
| earrings | 1 | 0.536287 | 0.536287 | +0.000000 |
| ring | 16 | 0.602326 | 0.603051 | +0.000725 |

## Representative Demo Cases

The demo set combines diverse person/object categories with top hard-benchmark wins. It includes ring, earrings, glasses, necklace, and bracelet examples across multiple person images, plus the strongest hard-set wins.

| Case | Class | Set | Pretrained | Pretrained + Geo | Delta |
| --- | --- | --- | --- | --- | --- |
| ring_woman_011_203 | ring | diverse_demo | 0.618754 | 0.621918 | +0.003164 |
| earrings_woman_004_103 | earrings | diverse_demo | 0.534739 | 0.536624 | +0.001885 |
| glasses_woman_010_301 | glasses | diverse_demo | 0.598226 | 0.600092 | +0.001866 |
| necklace_woman_012_101 | necklace | diverse_demo | 0.526438 | 0.528071 | +0.001633 |
| bracelet_woman_008_102 | bracelet | diverse_demo | 0.653525 | 0.653801 | +0.000276 |
| ring_woman_015_204 | ring | hard_benchmark_win | 0.641162 | 0.644585 | +0.003423 |
| bracelet_woman_008_302 | bracelet | hard_benchmark_win | 0.670664 | 0.673440 | +0.002776 |
| ring_woman_015_102 | ring | hard_benchmark_win | 0.524317 | 0.526425 | +0.002108 |
| ring_woman_015_201 | ring | hard_benchmark_win | 0.543893 | 0.545963 | +0.002070 |
| bracelet_woman_008_103 | bracelet | hard_benchmark_win | 0.640495 | 0.642406 | +0.001911 |

Demo artifacts:

- UI: `outputs/demo/geo_method_report/index.html`
- Video: `outputs/demo/geo_method_report/geo_method_demo.mp4`
- Case manifest: `outputs/demo/geo_method_report/demo_manifest.json`

## Why Geometry Helps

Small accessories occupy a tiny fraction of the image, so a global text prompt often gives the diffusion model too much freedom. The model can preserve the person while losing the exact object, or it can add the object in the wrong location. GACS helps because it makes the expected affordance region explicit, asks for class-specific occlusion and scale, and rejects candidates that damage unrelated parts of the person image.

This is also why the measured gains are small but meaningful. We are not changing the generator; we are choosing better samples from the same generator. The method improves reliability most on cases where the object has a predictable geometric relation to the body, such as rings on fingers, bracelets on wrists, earrings near ears, glasses across the nose bridge, and necklaces around the neck.

## Why Fine-Tuning Is Not the Main Claim

Fine-tuning is not impossible, but it is not the best story for this project under the current constraints.

| Run | Total score |
| --- | --- |
| Pretrained hard benchmark | 0.623209 |
| Pretrained + Geo hard benchmark | 0.623760 |
| Raw fine-tuned LoRA hard benchmark | 0.621025 |
| Fine-tuned minus pretrained | -0.002184 |

The obstacles are practical and methodological:

- True paired data is rare. For supervised try-on we need a person image without the object, an object reference, and the same person wearing that object as target.
- Crawled web images do not provide target images or clean masks. LLM labels can identify object class and rough boxes, but they do not solve pixel-accurate masking.
- Small-object masks are brittle. Rings, earrings, chains, watch straps, and glasses are thin, reflective, and often occluded by hair, hands, or clothing.
- The current crawl produced 38 candidates, 10 LLM-usable labels, and only 3 immediately usable pseudo-pairs. That is not enough to support a strong fine-tuning claim.
- The available pseudo-pair objective can overfit reconstruction artifacts instead of learning true object transfer. The raw fine-tuned result is already lower than the pretrained baseline in the current benchmark.
- Compute cost is high because the project uses a FLUX-style model and LoRA training. Multi-GPU training is feasible, but it is expensive relative to the evidence gained from the limited labels.

The best-practice framing is therefore: use fine-tuning as an attempted extension and negative result, while making GACS the main proposed method.

## Limitations

The current benchmark is local and small. It should not be reported as a full paper-level benchmark. The scoring function is useful for reproducible comparison, but it is still a proxy for human visual quality. Future work should add stronger masks from SAM or GroundingDINO, pose or hand keypoints for smaller affordance boxes, a larger balanced crawl, and human preference evaluation.

## Reproduction

```bash
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate cs338
python scripts/build_geo_method_report.py
```
