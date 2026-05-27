#!/usr/bin/env python
import argparse
import base64
import json
import mimetypes
import os
import re
import sys
from pathlib import Path

import requests

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from omnitry.enhance.cost_guard import BudgetExceeded, CostConfig, CostLedger, estimate_text_tokens, load_dotenv
from omnitry.enhance.data import load_json, write_json
from omnitry.enhance.taxonomy import HARD_CASE_CLASSES


SYSTEM_PROMPT = """You label images for virtual try-on small-object training.
Return only JSON that matches the schema. Do not guess when the object is unclear.
Focus on wearable or held try-on objects: ring, earrings, bracelet, necklace, watch,
glasses, sunglasses, bag, shoe, hat, belt, tie, bow tie, top clothes, bottom clothes, dress.
Reject images without a visible person, without a visible target object, with heavy blur,
or where the object cannot be localized for mask generation."""


def parse_args():
    parser = argparse.ArgumentParser(description="Label crawled try-on candidate images with an LLM and a hard cost cap.")
    parser.add_argument("--input", default="data/hard_cases/commons_hard_cases.json")
    parser.add_argument("--output", default="data/hard_cases/commons_llm_labels.json")
    parser.add_argument("--cost-state", default="outputs/llm_labeling/cost_state.json")
    parser.add_argument("--cost-events", default="outputs/llm_labeling/cost_events.jsonl")
    parser.add_argument("--model", default=os.environ.get("LLM_LABEL_MODEL", "gpt-4o-mini"))
    parser.add_argument("--api-base", default=os.environ.get("OPENAI_API_BASE", "https://api.openai.com/v1"))
    parser.add_argument("--max-items", type=int, default=None)
    parser.add_argument("--max-output-tokens", type=int, default=500)
    parser.add_argument("--image-detail", choices=["low", "high", "auto"], default=os.environ.get("LLM_LABEL_IMAGE_DETAIL", "low"))
    parser.add_argument("--dry-run", action="store_true", help="Estimate budget without sending API requests.")
    parser.add_argument("--skip-existing", action="store_true", default=True)
    parser.add_argument("--no-skip-existing", dest="skip_existing", action="store_false")
    return parser.parse_args()


def read_items(path):
    payload = load_json(Path(path))
    items = payload.get("items", payload) if isinstance(payload, dict) else payload
    if not isinstance(items, list):
        raise ValueError(f"Input must be a list or a dict with an items list: {path}")
    return payload, items


def existing_by_id(path):
    path = Path(path)
    if not path.is_file():
        return {}
    payload = load_json(path)
    return {item["id"]: item for item in payload.get("items", []) if "id" in item}


def resolve_path(value):
    if not value:
        return None
    value = str(value)
    if value.startswith("file://"):
        value = value[7:]
    path = Path(value)
    if path.is_absolute():
        return path
    return ROOT_DIR / path


def image_reference(item):
    for key in ("local_path", "image_path", "person_path"):
        value = item.get(key)
        if not value:
            continue
        path = resolve_path(value)
        if path and path.is_file():
            mime = mimetypes.guess_type(path.name)[0] or "image/jpeg"
            encoded = base64.b64encode(path.read_bytes()).decode("ascii")
            return f"data:{mime};base64,{encoded}", key
    for key in ("thumb_url", "url"):
        value = item.get(key)
        if value:
            return str(value), key
    raise FileNotFoundError(f"No usable image reference for item {item.get('id')}")


def user_prompt(item):
    category_hint = item.get("category") or item.get("query") or "unknown"
    title = item.get("title") or item.get("id") or ""
    license_name = item.get("license") or ""
    return f"""Label this image for virtual try-on training.

Candidate metadata:
- id: {item.get('id', '')}
- title: {title}
- category_hint: {category_hint}
- query: {item.get('query', '')}
- license: {license_name}

Return normalized bounding boxes in xyxy order with values from 0 to 1.
Use category \"none\" and usable=false if no good target object exists.
For each usable object, provide a short mask_prompt suitable for GroundingDINO/SAM."""


def label_schema():
    categories = sorted(HARD_CASE_CLASSES | {"belt", "top clothes", "bottom clothes", "dress", "none"})
    return {
        "type": "object",
        "additionalProperties": False,
        "required": ["usable", "category", "quality_score", "reason", "objects"],
        "properties": {
            "usable": {"type": "boolean"},
            "category": {"type": "string", "enum": categories},
            "quality_score": {"type": "number", "minimum": 0, "maximum": 1},
            "reason": {"type": "string"},
            "objects": {
                "type": "array",
                "maxItems": 5,
                "items": {
                    "type": "object",
                    "additionalProperties": False,
                    "required": [
                        "category",
                        "description",
                        "bbox_xyxy_norm",
                        "confidence",
                        "placement",
                        "occlusion",
                        "small_object",
                        "mask_prompt",
                        "reject_reason",
                    ],
                    "properties": {
                        "category": {"type": "string", "enum": categories},
                        "description": {"type": "string"},
                        "bbox_xyxy_norm": {
                            "type": "array",
                            "minItems": 4,
                            "maxItems": 4,
                            "items": {"type": "number", "minimum": 0, "maximum": 1},
                        },
                        "confidence": {"type": "number", "minimum": 0, "maximum": 1},
                        "placement": {"type": "string"},
                        "occlusion": {"type": "string"},
                        "small_object": {"type": "boolean"},
                        "mask_prompt": {"type": "string"},
                        "reject_reason": {"type": "string"},
                    },
                },
            },
        },
    }


def build_request(model, prompt, image_url, image_detail, max_output_tokens):
    return {
        "model": model,
        "temperature": 0,
        "max_completion_tokens": max_output_tokens,
        "response_format": {
            "type": "json_schema",
            "json_schema": {
                "name": "tryon_object_label",
                "strict": True,
                "schema": label_schema(),
            },
        },
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": prompt},
                    {"type": "image_url", "image_url": {"url": image_url, "detail": image_detail}},
                ],
            },
        ],
    }


def extract_json(text):
    text = (text or "").strip()
    fenced = re.search(r"```(?:json)?\s*(.*?)```", text, flags=re.S)
    if fenced:
        text = fenced.group(1).strip()
    return json.loads(text)


def normalize_label(label):
    label = dict(label or {})
    objects = [dict(item) for item in label.get("objects", []) if isinstance(item, dict)]
    usable_objects = [
        item
        for item in objects
        if item.get("category") and item.get("category") != "none" and float(item.get("confidence", 0.0) or 0.0) >= 0.3
    ]

    if label.get("usable") and usable_objects:
        if label.get("category") in {None, "", "none"}:
            label["category"] = usable_objects[0].get("category", "none")
            reason = label.get("reason", "")
            label["reason"] = (reason + " Root category normalized from the first usable object.").strip()
        label["objects"] = usable_objects
    elif label.get("usable"):
        label["usable"] = False
        label["category"] = "none"
        label["objects"] = objects
        reason = label.get("reason", "")
        label["reason"] = (reason + " Rejected because no usable localized object survived normalization.").strip()
    return label


def call_openai(api_base, api_key, payload, timeout=90):
    response = requests.post(
        f"{api_base.rstrip('/')}/chat/completions",
        headers={"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"},
        json=payload,
        timeout=timeout,
    )
    if response.status_code >= 400:
        raise RuntimeError(f"OpenAI API error {response.status_code}: {response.text[:1000]}")
    data = response.json()
    content = data["choices"][0]["message"]["content"]
    return normalize_label(extract_json(content)), data.get("usage", {})


def summarize_cost(items, config, max_output_tokens):
    total = 0.0
    estimates = []
    for item in items:
        prompt = user_prompt(item)
        input_tokens = estimate_text_tokens(SYSTEM_PROMPT + "\n" + prompt)
        cost = config.estimate(input_tokens, max_output_tokens, image_count=1)
        total += cost
        estimates.append({"id": item.get("id"), "estimated_cost_usd": round(cost, 6)})
    return total, estimates


def main():
    load_dotenv(ROOT_DIR / ".env")
    args = parse_args()
    input_payload, items = read_items(args.input)
    if args.max_items is not None:
        items = items[: args.max_items]

    config = CostConfig.from_env("LLM_LABEL_")
    ledger = CostLedger(args.cost_state, args.cost_events, config=config)
    existing = existing_by_id(args.output) if args.skip_existing else {}
    labels = dict(existing)

    pending = [item for item in items if item.get("id") not in labels]
    estimate_total, estimates = summarize_cost(pending, config, args.max_output_tokens)
    report = {
        "model": args.model,
        "pending_items": len(pending),
        "estimated_new_cost_usd": round(estimate_total, 6),
        "ledger_spent_usd": round(ledger.spent_usd, 6),
        "soft_budget_usd": config.soft_budget_usd,
        "budget_usd": config.budget_usd,
        "remaining_to_soft_budget_usd": round(ledger.remaining_to_soft_budget(), 6),
        "price_config": {
            "input_usd_per_1m": config.input_usd_per_1m,
            "output_usd_per_1m": config.output_usd_per_1m,
            "request_overhead_usd": config.request_overhead_usd,
            "image_input_tokens": config.image_input_tokens,
        },
        "items": estimates,
    }
    if args.dry_run:
        print(json.dumps(report, indent=2))
        return 0

    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        raise RuntimeError("OPENAI_API_KEY is missing. Put it in .env or export it before running labels.")

    output_items = list(labels.values())
    blocked = False
    for item in pending:
        item_id = item.get("id") or f"item_{len(output_items):06d}"
        prompt = user_prompt(item)
        image_url, image_source = image_reference(item)
        input_tokens = estimate_text_tokens(SYSTEM_PROMPT + "\n" + prompt)
        estimate = config.estimate(input_tokens, args.max_output_tokens, image_count=1)
        metadata = {"id": item_id, "image_source": image_source, "model": args.model}
        try:
            ledger.reserve(item_id, estimate, metadata=metadata)
        except BudgetExceeded as exc:
            print(str(exc))
            blocked = True
            break

        try:
            payload = build_request(args.model, prompt, image_url, args.image_detail, args.max_output_tokens)
            label, usage = call_openai(args.api_base, api_key, payload)
            ledger.complete(item_id, usage=usage, metadata=metadata)
        except Exception as exc:
            ledger.fail(item_id, str(exc), metadata=metadata)
            raise

        output_items.append(
            {
                **item,
                "llm_label": label,
                "llm_label_model": args.model,
                "llm_label_estimated_cost_usd": round(estimate, 6),
            }
        )
        write_json(
            Path(args.output),
            {
                "source": input_payload.get("source", "crawled_images") if isinstance(input_payload, dict) else "crawled_images",
                "label_model": args.model,
                "cost_state": str(args.cost_state),
                "complete": False,
                "items": output_items,
            },
        )
        print(f"labeled {item_id}: usable={label.get('usable')} category={label.get('category')} spent=${ledger.spent_usd:.4f}")

    write_json(
        Path(args.output),
        {
            "source": input_payload.get("source", "crawled_images") if isinstance(input_payload, dict) else "crawled_images",
            "label_model": args.model,
            "cost_state": str(args.cost_state),
            "complete": not blocked and len(output_items) >= len(items),
            "blocked_by_budget": blocked,
            "items": output_items,
        },
    )
    print(f"Wrote labels -> {args.output}")
    print(f"Cost ledger -> {args.cost_state} spent=${ledger.spent_usd:.4f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
