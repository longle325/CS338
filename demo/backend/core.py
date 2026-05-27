from __future__ import annotations

import json
import os
import random
import sys
import threading
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable

from PIL import Image

ROOT_DIR = Path(__file__).resolve().parents[2]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

import gradio_demo as demo
from omnitry.enhance import CandidateResult, confidence_label, format_diagnostics, score_candidate

OUTPUT_ROOT = Path(os.environ.get("OMNITRY_BACKEND_OUTPUT_ROOT", ROOT_DIR / "outputs/live_demo_backend"))
ARTIFACT_ROUTE = os.environ.get("OMNITRY_BACKEND_ARTIFACT_ROUTE", "/artifacts").rstrip("/")
MAX_CANDIDATES = max(1, int(os.environ.get("OMNITRY_BACKEND_MAX_CANDIDATES", demo.MAX_CANDIDATES)))

CLASS_ALIASES = {
    "eyeglasses": "glasses",
    "glass": "glasses",
    "sunglass": "sunglasses",
    "earring": "earrings",
    "shoe": "shoe",
    "shoes": "shoe",
    "bracelets": "bracelet",
    "rings": "ring",
    "necklaces": "necklace",
    "top": "top clothes",
    "top cloth": "top clothes",
    "shirt": "top clothes",
    "bottom": "bottom clothes",
    "bottom cloth": "bottom clothes",
    "pants": "bottom clothes",
    "bowtie": "bow tie",
}

FILE_SLUGS = {
    "top clothes": "top_cloth",
    "bottom clothes": "bottom_cloth",
    "dress": "dress",
    "shoe": "shoes",
    "earrings": "earrings",
    "bracelet": "bracelet",
    "necklace": "necklace",
    "ring": "ring",
    "sunglasses": "sunglasses",
    "glasses": "glasses",
    "belt": "belt",
    "bag": "bag",
    "hat": "hat",
    "tie": "tie",
    "bow tie": "bowtie",
}

INFERENCE_LOCK = threading.Lock()


@dataclass(frozen=True)
class CompareSettings:
    object_class: str
    optional_prompt: str = ""
    steps: int = 20
    guidance_scale: float = 30.0
    seed: int = -1
    geometry_candidate_count: int = 2
    run_pretrained: bool = True
    run_geometry: bool = True


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_suffix(path.suffix + ".tmp")
    tmp_path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    tmp_path.replace(path)


def read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def artifact_url(run_id: str, relative_path: str | Path) -> str:
    rel = str(relative_path).replace(os.sep, "/").lstrip("/")
    return f"{ARTIFACT_ROUTE}/{run_id}/{rel}"


def normalize_object_class(raw: str) -> str:
    key = (raw or "").strip().lower().replace("_", " ")
    key = " ".join(key.split())
    object_map = {str(name).lower(): str(name) for name in demo.args.object_map.keys()}
    if key in object_map:
        return object_map[key]
    alias = CLASS_ALIASES.get(key)
    if alias and alias in demo.args.object_map:
        return alias
    raise ValueError(f"Unknown object_class '{raw}'. Use one of: {', '.join(object_classes())}")


def object_classes() -> list[str]:
    return list(demo.args.object_map.keys())


def example_items() -> list[dict]:
    items = []
    example_dir = ROOT_DIR / "demo_example"
    for object_class in object_classes():
        slug = FILE_SLUGS.get(object_class, object_class.replace(" ", "_"))
        person_path = example_dir / f"person_{slug}.jpg"
        object_path = example_dir / f"object_{slug}.jpg"
        if not person_path.is_file() or not object_path.is_file():
            continue
        items.append(
            {
                "id": slug,
                "object_class": object_class,
                "person_url": f"/demo-examples/person_{slug}.jpg",
                "object_url": f"/demo-examples/object_{slug}.jpg",
            }
        )
    return items


def clamp_settings(settings: CompareSettings) -> CompareSettings:
    object_class = normalize_object_class(settings.object_class)
    steps = max(1, min(50, int(settings.steps)))
    guidance_scale = max(1.0, min(50.0, float(settings.guidance_scale)))
    candidate_count = max(1, min(MAX_CANDIDATES, int(settings.geometry_candidate_count)))
    seed = int(settings.seed)
    if seed < 0:
        seed = random.randint(0, 2**32 - 1)
    seed = seed % (2**32)
    return CompareSettings(
        object_class=object_class,
        optional_prompt=(settings.optional_prompt or "").strip(),
        steps=steps,
        guidance_scale=guidance_scale,
        seed=seed,
        geometry_candidate_count=candidate_count,
        run_pretrained=bool(settings.run_pretrained),
        run_geometry=bool(settings.run_geometry),
    )


def load_rgb(path: Path) -> Image.Image:
    with Image.open(path) as image:
        return image.convert("RGB")


def score_payload(total: float, object_score: float, person_score: float, artifact_score: float) -> dict:
    return {
        "total": round(float(total), 6),
        "object": round(float(object_score), 6),
        "person": round(float(person_score), 6),
        "artifact": round(float(artifact_score), 6),
        "confidence": confidence_label(float(total)),
    }


def candidate_payload(
    run_id: str,
    branch_name: str,
    candidate: CandidateResult,
    rank: int,
    selected: bool,
    image_path: Path,
    run_dir: Path,
) -> dict:
    relative_path = image_path.relative_to(run_dir)
    return {
        "rank": rank,
        "seed": int(candidate.seed),
        "selected": selected,
        "image_path": str(image_path),
        "image_url": artifact_url(run_id, relative_path),
        "branch": branch_name,
        "scores": score_payload(
            candidate.score,
            candidate.object_score,
            candidate.person_score,
            candidate.artifact_score,
        ),
    }


def run_branch(
    *,
    run_id: str,
    run_dir: Path,
    branch_name: str,
    mode: str,
    person_image: Image.Image,
    object_image: Image.Image,
    object_class: str,
    optional_prompt: str,
    steps: int,
    guidance_scale: float,
    seed: int,
    candidate_count: int,
    update: Callable[[str, str], None],
) -> dict:
    update(f"running_{branch_name}", f"Generating {branch_name} output")
    prompt = demo.build_prompt(object_class, optional_prompt, mode)
    effective_candidate_count = candidate_count if mode == "Enhanced" else 1

    person_image, object_image, person_tensor, object_image_padded, target_width, target_height = demo.prepare_condition_tensors(
        person_image,
        object_image,
    )

    candidates: list[CandidateResult] = []
    seeds = demo._candidate_seeds(seed, effective_candidate_count)
    candidate_dir = run_dir / "candidates" / branch_name
    candidate_dir.mkdir(parents=True, exist_ok=True)

    for index, candidate_seed in enumerate(seeds):
        update(
            f"running_{branch_name}",
            f"Generating {branch_name} candidate {index + 1}/{len(seeds)}",
        )
        image = demo._run_candidate(
            person_tensor,
            object_image_padded,
            prompt,
            steps,
            guidance_scale,
            candidate_seed,
            target_width,
            target_height,
        )
        total, object_score, person_score, artifact_score = score_candidate(
            image,
            person_image,
            object_image,
            object_class,
        )
        candidate = CandidateResult(image, candidate_seed, total, object_score, person_score, artifact_score)
        candidates.append(candidate)
        image.save(candidate_dir / f"candidate_{index}.jpg", quality=95)

    best_index = max(range(len(candidates)), key=lambda item_index: candidates[item_index].score)
    best = candidates[best_index]
    output_path = run_dir / f"{branch_name}.jpg"
    diagnostics_path = run_dir / f"{branch_name}_diagnostics.md"
    best.image.save(output_path, quality=95)
    diagnostics = format_diagnostics(prompt, candidates, best_index, mode=mode)
    diagnostics_path.write_text(diagnostics, encoding="utf-8")

    ranked_candidates = sorted(enumerate(candidates), key=lambda pair: pair[1].score, reverse=True)
    candidate_items = []
    for rank, (candidate_index, candidate) in enumerate(ranked_candidates, 1):
        candidate_items.append(
            candidate_payload(
                run_id,
                branch_name,
                candidate,
                rank,
                selected=candidate_index == best_index,
                image_path=candidate_dir / f"candidate_{candidate_index}.jpg",
                run_dir=run_dir,
            )
        )

    return {
        "label": "Pretrained + Geometry" if mode == "Enhanced" else "Pretrained",
        "branch": branch_name,
        "mode": mode,
        "prompt": prompt,
        "selected_seed": int(best.seed),
        "candidate_count": len(candidates),
        "image_path": str(output_path),
        "image_url": artifact_url(run_id, output_path.relative_to(run_dir)),
        "diagnostics_path": str(diagnostics_path),
        "diagnostics_url": artifact_url(run_id, diagnostics_path.relative_to(run_dir)),
        "scores": score_payload(best.score, best.object_score, best.person_score, best.artifact_score),
        "candidates": candidate_items,
    }


def delta_payload(pretrained: dict | None, geometry: dict | None) -> dict | None:
    if not pretrained or not geometry:
        return None
    keys = ["total", "object", "person", "artifact"]
    delta = {
        key: round(float(geometry["scores"][key]) - float(pretrained["scores"][key]), 6)
        for key in keys
    }
    delta["winner"] = "geometry" if delta["total"] > 0 else "pretrained" if delta["total"] < 0 else "tie"
    delta["reason"] = selection_reason(delta)
    return delta


def selection_reason(delta: dict) -> str:
    if delta["total"] <= 0:
        return "Geometry did not improve the total score for this input."
    if delta["object"] > 0 and delta["person"] >= -0.03:
        return "Geometry improved object preservation while keeping person preservation stable."
    if delta["object"] > 0:
        return "Geometry selected the candidate with stronger target-object evidence."
    if delta["artifact"] > 0:
        return "Geometry selected the cleaner candidate with fewer visual artifacts."
    return "Geometry improved the weighted selection score."


def run_compare(
    *,
    run_id: str,
    run_dir: Path,
    settings: CompareSettings,
    update: Callable[[str, str], None] | None = None,
) -> dict:
    def notify(stage: str, message: str) -> None:
        if update:
            update(stage, message)

    settings = clamp_settings(settings)
    started = time.time()
    person_path = run_dir / "person.jpg"
    object_path = run_dir / "object.jpg"
    result_path = run_dir / "result.json"

    notify("waiting_for_inference_lock", "Waiting for the shared inference lock")
    with INFERENCE_LOCK:
        notify("loading_model", "Validating checkpoints and loading the OmniTry pipeline if needed")
        demo.validate_checkpoint_paths()
        person_image = load_rgb(person_path)
        object_image = load_rgb(object_path)

        pretrained = None
        geometry = None
        if settings.run_pretrained:
            pretrained = run_branch(
                run_id=run_id,
                run_dir=run_dir,
                branch_name="pretrained",
                mode="Baseline",
                person_image=person_image,
                object_image=object_image,
                object_class=settings.object_class,
                optional_prompt=settings.optional_prompt,
                steps=settings.steps,
                guidance_scale=settings.guidance_scale,
                seed=settings.seed,
                candidate_count=1,
                update=notify,
            )
        if settings.run_geometry:
            geometry = run_branch(
                run_id=run_id,
                run_dir=run_dir,
                branch_name="geometry",
                mode="Enhanced",
                person_image=person_image,
                object_image=object_image,
                object_class=settings.object_class,
                optional_prompt=settings.optional_prompt,
                steps=settings.steps,
                guidance_scale=settings.guidance_scale,
                seed=settings.seed,
                candidate_count=settings.geometry_candidate_count,
                update=notify,
            )

    result = {
        "run_id": run_id,
        "created_at": utc_now(),
        "elapsed_seconds": round(time.time() - started, 3),
        "settings": settings.__dict__,
        "inputs": {
            "person_path": str(person_path),
            "person_url": artifact_url(run_id, "person.jpg"),
            "object_path": str(object_path),
            "object_url": artifact_url(run_id, "object.jpg"),
            "object_class": settings.object_class,
        },
        "pretrained": pretrained,
        "geometry": geometry,
        "delta": delta_payload(pretrained, geometry),
    }
    write_json(result_path, result)
    return result
