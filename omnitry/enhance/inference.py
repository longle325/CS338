from dataclasses import dataclass
from typing import List, Optional, Tuple

import numpy as np

from .taxonomy import AFFORDANCE_BOXES, AFFORDANCE_PROMPTS, normalize_class_name


@dataclass
class CandidateResult:
    image: object
    seed: int
    score: float
    object_score: float
    person_score: float
    artifact_score: float


def build_enhanced_prompt(base_prompt: str, object_class: str, optional_prompt: Optional[str] = "", enabled: bool = True) -> str:
    pieces = [base_prompt]
    if enabled:
        normalized_class = normalize_class_name(object_class)
        pieces.extend(
            [
                AFFORDANCE_PROMPTS.get(
                    normalized_class,
                    "Place the object naturally on the person with realistic scale, pose, and occlusion.",
                ),
                "Preserve the object identity, color, material, texture, logo, silhouette, and small details.",
                "Preserve the person identity, face, hands, skin, hair, body shape, background, and all unrelated clothing.",
            ]
        )

    optional_prompt = (optional_prompt or "").strip()
    if optional_prompt:
        pieces.append(optional_prompt)

    return " ".join(piece.strip() for piece in pieces if piece and piece.strip())


def _normalized_box(object_class, width, height):
    normalized_class = normalize_class_name(object_class)
    x1, y1, x2, y2 = AFFORDANCE_BOXES.get(normalized_class, (0.0, 0.0, 1.0, 1.0))
    return (
        max(0, min(width - 1, int(x1 * width))),
        max(0, min(height - 1, int(y1 * height))),
        max(1, min(width, int(x2 * width))),
        max(1, min(height, int(y2 * height))),
    )


def _array_rgb(image, size=None):
    if size is not None:
        image = image.resize(size)
    return np.asarray(image.convert("RGB"), dtype=np.float32) / 255.0


def _foreground_pixels(image):
    arr = _array_rgb(image, size=(192, 192))
    corners = np.concatenate(
        [
            arr[:8, :8].reshape(-1, 3),
            arr[:8, -8:].reshape(-1, 3),
            arr[-8:, :8].reshape(-1, 3),
            arr[-8:, -8:].reshape(-1, 3),
        ],
        axis=0,
    )
    bg = np.median(corners, axis=0)
    distance = np.linalg.norm(arr - bg, axis=-1)
    pixels = arr[distance > 0.08]
    if pixels.shape[0] < 256:
        pixels = arr.reshape(-1, 3)
    return pixels


def _rgb_histogram(pixels, bins=8):
    hist, _ = np.histogramdd(
        pixels.reshape(-1, 3),
        bins=(bins, bins, bins),
        range=((0, 1), (0, 1), (0, 1)),
    )
    hist = hist.astype(np.float32)
    hist /= hist.sum() + 1e-8
    return hist.reshape(-1)


def _histogram_intersection(hist_a, hist_b):
    return float(np.minimum(hist_a, hist_b).sum())


def _crop_affordance(image, object_class):
    width, height = image.size
    box = _normalized_box(object_class, width, height)
    return image.crop(box)


def _mask_out_box(shape, box):
    height, width = shape[:2]
    x1, y1, x2, y2 = box
    mask = np.ones((height, width), dtype=bool)
    mask[y1:y2, x1:x2] = False
    return mask


def score_candidate(candidate_image, person_image, object_image, object_class) -> Tuple[float, float, float, float]:
    object_hist = _rgb_histogram(_foreground_pixels(object_image))
    target_crop = _crop_affordance(candidate_image, object_class)
    target_hist = _rgb_histogram(_array_rgb(target_crop, size=(192, 192)).reshape(-1, 3))
    object_score = _histogram_intersection(object_hist, target_hist)

    score_size = (256, 256)
    person_arr = _array_rgb(person_image, size=score_size)
    candidate_arr = _array_rgb(candidate_image, size=score_size)
    box = _normalized_box(object_class, score_size[0], score_size[1])
    outside = _mask_out_box(candidate_arr.shape, box)
    diff = np.abs(person_arr[outside] - candidate_arr[outside]).mean()
    person_score = float(np.clip(1.0 - diff / 0.35, 0.0, 1.0))

    gray = candidate_arr.mean(axis=2)
    grad_y, grad_x = np.gradient(gray)
    sharpness = np.sqrt(grad_x**2 + grad_y**2).mean()
    sharpness_score = float(np.clip((sharpness - 0.015) / 0.08, 0.0, 1.0))
    contrast_score = float(np.clip(gray.std() / 0.22, 0.0, 1.0))
    extreme_fraction = float(((candidate_arr < 0.02) | (candidate_arr > 0.98)).mean())
    pixel_health = float(np.clip(1.0 - max(0.0, extreme_fraction - 0.08) / 0.35, 0.0, 1.0))
    artifact_score = 0.40 * sharpness_score + 0.35 * contrast_score + 0.25 * pixel_health

    total = 0.35 * object_score + 0.35 * person_score + 0.30 * artifact_score
    return float(total), float(object_score), float(person_score), float(artifact_score)


def confidence_label(score):
    if score >= 0.72:
        return "high"
    if score >= 0.55:
        return "medium"
    return "low"


def format_diagnostics(prompt, candidates: List[CandidateResult], best_index, mode="Enhanced"):
    best = candidates[best_index]
    lines = [
        f"**Mode:** {mode}  ",
        f"**Confidence:** {confidence_label(best.score)}  ",
        f"**Best seed:** `{best.seed}`  ",
        f"**Prompt used:** {prompt}",
        "",
        "| rank | seed | total | object | person | artifact |",
        "|---:|---:|---:|---:|---:|---:|",
    ]
    ranked = sorted(candidates, key=lambda item: item.score, reverse=True)
    for rank, item in enumerate(ranked, 1):
        lines.append(
            f"| {rank} | {item.seed} | {item.score:.3f} | {item.object_score:.3f} | "
            f"{item.person_score:.3f} | {item.artifact_score:.3f} |"
        )
    return "\n".join(lines)
