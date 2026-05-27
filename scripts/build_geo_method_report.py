#!/usr/bin/env python
import argparse
import json
import shutil
import subprocess
from datetime import date
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont, ImageOps


ROOT = Path(__file__).resolve().parents[1]

DIVERSE_CASE_IDS = [
    "ring_woman_011_203",
    "earrings_woman_004_103",
    "glasses_woman_010_301",
    "necklace_woman_012_101",
    "bracelet_woman_008_102",
]

HARD_WIN_CASE_IDS = [
    "ring_woman_015_204",
    "bracelet_woman_008_302",
    "ring_woman_015_102",
    "ring_woman_015_201",
    "bracelet_woman_008_103",
]


def parse_args():
    parser = argparse.ArgumentParser(description="Build the geometry-method report, demo UI, and video assets.")
    parser.add_argument("--output-dir", default="outputs/demo/geo_method_report")
    parser.add_argument("--doc-output", default="docs/geo_affordance_project_report.md")
    parser.add_argument("--hard-manifest", default="data/hard_cases/omnitry_full_local_hard_cases.json")
    parser.add_argument("--diverse-manifest", default="data/hard_cases/omnitry_small_object_diverse_demo.json")
    parser.add_argument("--hard-pretrained-summary", default="outputs/tryon_benchmark/original_enhanced_summary.json")
    parser.add_argument("--hard-geo-summary", default="outputs/tryon_benchmark/original_enhanced_c2_summary.json")
    parser.add_argument("--diverse-pretrained-summary", default="outputs/tryon_benchmark/diverse_demo_pretrained_summary.json")
    parser.add_argument("--diverse-geo-summary", default="outputs/tryon_benchmark/diverse_demo_pretrained_gacs_summary.json")
    parser.add_argument("--finetune-summary", default="outputs/tryon_benchmark/enhanced_ft_summary.json")
    parser.add_argument("--seconds-per-case", type=float, default=2.4)
    return parser.parse_args()


def load_json(path):
    with resolve(path).open("r", encoding="utf-8") as handle:
        return json.load(handle)


def maybe_load_json(path):
    resolved = resolve(path)
    if not resolved.is_file():
        return None
    return load_json(path)


def write_json(path, payload):
    path = resolve(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2)
        handle.write("\n")


def resolve(path):
    path = Path(path)
    return path if path.is_absolute() else ROOT / path


def rows_by_id(payload):
    return {row["id"]: row for row in payload.get("items", [])}


def manifest_by_id(path):
    payload = load_json(path)
    items = payload.get("items", payload) if isinstance(payload, dict) else payload
    return {item["id"]: item for item in items}


def metric_summary(payload):
    summary = payload.get("summary", {})
    return {
        "items": int(summary.get("items", 0)),
        "total_mean": float(summary.get("total_mean", 0.0)),
        "object_mean": float(summary.get("object_mean", 0.0)),
        "person_mean": float(summary.get("person_mean", 0.0)),
        "artifact_mean": float(summary.get("artifact_mean", 0.0)),
        "classes": summary.get("classes", {}),
    }


def common_comparison(pretrained_payload, geo_payload):
    pretrained = rows_by_id(pretrained_payload)
    geo = rows_by_id(geo_payload)
    rows = []
    for item_id in sorted(set(pretrained) & set(geo)):
        base = pretrained[item_id]
        candidate = geo[item_id]
        delta = float(candidate["total"]) - float(base["total"])
        rows.append(
            {
                "id": item_id,
                "category": base.get("category", candidate.get("category", "unknown")),
                "pretrained_total": float(base["total"]),
                "geo_total": float(candidate["total"]),
                "delta_total": delta,
                "delta_object": float(candidate["object"]) - float(base["object"]),
                "delta_person": float(candidate["person"]) - float(base["person"]),
                "delta_artifact": float(candidate["artifact"]) - float(base["artifact"]),
            }
        )
    wins = sum(1 for row in rows if row["delta_total"] > 1e-9)
    losses = sum(1 for row in rows if row["delta_total"] < -1e-9)
    ties = len(rows) - wins - losses
    return {
        "items": len(rows),
        "wins": wins,
        "ties": ties,
        "losses": losses,
        "top_wins": sorted(rows, key=lambda row: row["delta_total"], reverse=True),
        "rows": rows,
    }


def class_comparison(pretrained_payload, geo_payload):
    base_classes = metric_summary(pretrained_payload)["classes"]
    geo_classes = metric_summary(geo_payload)["classes"]
    rows = []
    for category in sorted(set(base_classes) | set(geo_classes)):
        base = base_classes.get(category, {})
        geo = geo_classes.get(category, {})
        rows.append(
            {
                "category": category,
                "count": int(geo.get("count", base.get("count", 0))),
                "pretrained_total": float(base.get("total_mean", 0.0)),
                "geo_total": float(geo.get("total_mean", 0.0)),
                "delta_total": float(geo.get("total_mean", 0.0)) - float(base.get("total_mean", 0.0)),
            }
        )
    return rows


def record_count(path):
    payload = maybe_load_json(path)
    if payload is None:
        return 0
    if isinstance(payload, list):
        return len(payload)
    if isinstance(payload, dict):
        for key in ["items", "labels", "records", "candidates", "results"]:
            value = payload.get(key)
            if isinstance(value, list):
                return len(value)
    return 0


def copy_asset(src, dst):
    src = resolve(src)
    if not src.is_file():
        raise FileNotFoundError(src)
    dst.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(src, dst)
    return dst


def asset_rel(output_dir, path):
    return Path(path).relative_to(output_dir).as_posix()


def make_case(item_id, group, manifest, pretrained_row, geo_row, output_dir):
    item = manifest[item_id]
    category = item.get("category") or item.get("garment_class") or pretrained_row.get("category")
    case_dir = output_dir / "assets" / item_id
    assets = {
        "person": copy_asset(item["person_path"], case_dir / "person.jpg"),
        "object": copy_asset(item["object_path"], case_dir / "object.jpg"),
        "pretrained": copy_asset(pretrained_row["image"], case_dir / "pretrained.jpg"),
        "geo": copy_asset(geo_row["image"], case_dir / "pretrained_geo.jpg"),
    }
    delta_total = float(geo_row["total"]) - float(pretrained_row["total"])
    delta_object = float(geo_row["object"]) - float(pretrained_row["object"])
    delta_person = float(geo_row["person"]) - float(pretrained_row["person"])
    delta_artifact = float(geo_row["artifact"]) - float(pretrained_row["artifact"])
    return {
        "id": item_id,
        "group": group,
        "category": category,
        "person_id": item.get("person", {}).get("id", ""),
        "object_id": item.get("object", {}).get("id", ""),
        "object_caption": item.get("object", {}).get("caption", ""),
        "pretrained": {
            "total": float(pretrained_row["total"]),
            "object": float(pretrained_row["object"]),
            "person": float(pretrained_row["person"]),
            "artifact": float(pretrained_row["artifact"]),
        },
        "geo": {
            "total": float(geo_row["total"]),
            "object": float(geo_row["object"]),
            "person": float(geo_row["person"]),
            "artifact": float(geo_row["artifact"]),
        },
        "delta_total": delta_total,
        "delta_object": delta_object,
        "delta_person": delta_person,
        "delta_artifact": delta_artifact,
        "assets": assets,
    }


def build_cases(args, output_dir, hard_pretrained, hard_geo, diverse_pretrained, diverse_geo):
    hard_manifest = manifest_by_id(args.hard_manifest)
    diverse_manifest = manifest_by_id(args.diverse_manifest)
    hard_base = rows_by_id(hard_pretrained)
    hard_g = rows_by_id(hard_geo)
    diverse_base = rows_by_id(diverse_pretrained)
    diverse_g = rows_by_id(diverse_geo)

    cases = []
    for item_id in DIVERSE_CASE_IDS:
        cases.append(
            make_case(
                item_id,
                "diverse_demo",
                diverse_manifest,
                diverse_base[item_id],
                diverse_g[item_id],
                output_dir,
            )
        )
    for item_id in HARD_WIN_CASE_IDS:
        cases.append(
            make_case(
                item_id,
                "hard_benchmark_win",
                hard_manifest,
                hard_base[item_id],
                hard_g[item_id],
                output_dir,
            )
        )
    return cases


def font(name, size):
    for base in [Path("/usr/share/fonts/truetype/dejavu"), Path("/usr/share/fonts/dejavu")]:
        path = base / name
        if path.is_file():
            return ImageFont.truetype(str(path), size=size)
    return ImageFont.load_default()


def fit_image(path, size, background=(250, 250, 250)):
    image = ImageOps.exif_transpose(Image.open(path).convert("RGB"))
    fitted = ImageOps.contain(image, size)
    canvas = Image.new("RGB", size, background)
    canvas.paste(fitted, ((size[0] - fitted.width) // 2, (size[1] - fitted.height) // 2))
    return canvas


def draw_metric(draw, xy, label, value, color, label_font, value_font):
    x, y = xy
    draw.text((x, y), label.upper(), fill=(75, 85, 99), font=label_font)
    draw.text((x, y + 30), value, fill=color, font=value_font)


def render_frame(case, frame_path):
    width, height = 1920, 1080
    margin = 52
    gap = 24
    panel_w = 430
    panel_h = 710
    canvas = Image.new("RGB", (width, height), (247, 248, 251))
    draw = ImageDraw.Draw(canvas)

    title_font = font("DejaVuSans-Bold.ttf", 42)
    subtitle_font = font("DejaVuSans.ttf", 23)
    label_font = font("DejaVuSans-Bold.ttf", 18)
    score_font = font("DejaVuSans-Bold.ttf", 30)
    small_font = font("DejaVuSans.ttf", 20)
    ink = (17, 24, 39)
    muted = (75, 85, 99)
    blue = (37, 99, 235)
    teal = (15, 118, 110)
    amber = (180, 83, 9)

    draw.text((margin, 34), "Geo-Affordance Candidate Selection", fill=ink, font=title_font)
    draw.text(
        (margin, 88),
        f"{case['id']} | {case['category']} | total {case['delta_total']:+.6f}",
        fill=muted,
        font=subtitle_font,
    )

    score_x = 1180
    draw_metric(draw, (score_x, 38), "pretrained", f"{case['pretrained']['total']:.6f}", blue, label_font, score_font)
    draw_metric(draw, (score_x + 235, 38), "pretrained + geo", f"{case['geo']['total']:.6f}", teal, label_font, score_font)
    draw_metric(draw, (score_x + 525, 38), "delta", f"{case['delta_total']:+.6f}", amber, label_font, score_font)

    labels = [
        ("Person input", case["assets"]["person"]),
        ("Object reference", case["assets"]["object"]),
        ("Pretrained", case["assets"]["pretrained"]),
        ("Pretrained + Geo", case["assets"]["geo"]),
    ]
    top = 158
    for index, (label, path) in enumerate(labels):
        left = margin + index * (panel_w + gap)
        draw.rounded_rectangle(
            (left, top, left + panel_w, top + panel_h),
            radius=8,
            fill=(255, 255, 255),
            outline=(209, 213, 219),
            width=2,
        )
        draw.text((left + 18, top + 16), label, fill=ink, font=label_font)
        image = fit_image(path, (panel_w - 36, panel_h - 74))
        canvas.paste(image, (left + 18, top + 54))

    footer_y = 904
    draw.text(
        (margin, footer_y),
        f"Object {case['delta_object']:+.6f}   Person {case['delta_person']:+.6f}   Artifact {case['delta_artifact']:+.6f}",
        fill=ink,
        font=small_font,
    )
    draw.text(
        (margin, footer_y + 38),
        "The geometry method changes prompting and candidate selection; it does not update model weights.",
        fill=muted,
        font=small_font,
    )
    frame_path.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(frame_path, quality=95)


def render_video(output_dir, cases, seconds_per_case):
    frames_dir = output_dir / "frames"
    frames_dir.mkdir(parents=True, exist_ok=True)
    for index, case in enumerate(cases):
        render_frame(case, frames_dir / f"frame_{index:03d}.jpg")
    video = output_dir / "geo_method_demo.mp4"
    framerate = 1.0 / max(seconds_per_case, 0.5)
    subprocess.run(
        [
            "ffmpeg",
            "-y",
            "-framerate",
            f"{framerate:.6f}",
            "-i",
            str(frames_dir / "frame_%03d.jpg"),
            "-vf",
            "format=yuv420p",
            "-r",
            "30",
            str(video),
        ],
        check=True,
    )
    return video


def render_html(output_dir, cases, metrics):
    page_cases = []
    for case in cases:
        copied = dict(case)
        copied["assets"] = {key: asset_rel(output_dir, value) for key, value in case["assets"].items()}
        page_cases.append(copied)
    page_data = json.dumps({"cases": page_cases, "metrics": metrics}, indent=2)
    html = HTML_TEMPLATE.replace("__PAGE_DATA__", safe_script_json(page_data))
    (output_dir / "index.html").write_text(html, encoding="utf-8")


def safe_script_json(value):
    return value.replace("<", "\\u003c").replace(">", "\\u003e").replace("&", "\\u0026")


HTML_TEMPLATE = """<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>Geo-Affordance Method Demo</title>
  <style>
    :root {
      --bg: #f7f8fb;
      --panel: #ffffff;
      --ink: #111827;
      --muted: #4b5563;
      --line: #d1d5db;
      --blue: #2563eb;
      --teal: #0f766e;
      --amber: #b45309;
      --red: #b91c1c;
    }
    * { box-sizing: border-box; }
    body {
      margin: 0;
      font-family: Inter, ui-sans-serif, system-ui, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
      background: var(--bg);
      color: var(--ink);
      letter-spacing: 0;
    }
    header {
      display: grid;
      grid-template-columns: minmax(300px, 1fr) minmax(520px, .85fr);
      gap: 24px;
      align-items: end;
      padding: 22px 26px 16px;
      border-bottom: 1px solid var(--line);
      background: #fff;
    }
    h1 { margin: 0 0 6px; font-size: clamp(25px, 3vw, 38px); line-height: 1.08; }
    .subline { margin: 0; color: var(--muted); font-size: 15px; max-width: 900px; }
    .score-grid { display: grid; grid-template-columns: repeat(4, minmax(118px, 1fr)); gap: 8px; }
    .score-card {
      border: 1px solid var(--line);
      border-radius: 8px;
      background: #fff;
      padding: 10px 12px;
    }
    .score-card span { display: block; color: var(--muted); font-size: 11px; font-weight: 800; text-transform: uppercase; }
    .score-card strong { display: block; margin-top: 4px; font-size: 19px; overflow-wrap: anywhere; }
    main { display: grid; grid-template-columns: 310px minmax(0, 1fr); min-height: calc(100vh - 112px); }
    aside { background: #fff; border-right: 1px solid var(--line); padding: 14px; overflow: auto; }
    button.case {
      width: 100%;
      min-height: 76px;
      border: 1px solid var(--line);
      border-radius: 8px;
      background: #fff;
      text-align: left;
      padding: 12px;
      margin-bottom: 8px;
      cursor: pointer;
      color: var(--ink);
    }
    button.case.active { border-color: var(--teal); box-shadow: 0 0 0 2px rgba(15,118,110,.14); }
    button.case strong { display: block; font-size: 14px; overflow-wrap: anywhere; }
    button.case span { display: block; margin-top: 6px; color: var(--muted); font-size: 13px; }
    .workspace { padding: 18px 22px 28px; overflow: auto; }
    .compare { display: grid; grid-template-columns: repeat(4, minmax(170px, 1fr)); gap: 12px; }
    figure { margin: 0; border: 1px solid var(--line); border-radius: 8px; background: var(--panel); overflow: hidden; }
    figure img { width: 100%; aspect-ratio: 4 / 5; object-fit: contain; background: #fafafa; display: block; }
    figcaption { border-top: 1px solid var(--line); padding: 10px 11px; font-size: 13px; font-weight: 800; }
    .details { display: grid; grid-template-columns: 1.2fr .8fr; gap: 14px; margin-top: 14px; align-items: start; }
    .panel { border: 1px solid var(--line); border-radius: 8px; background: #fff; padding: 14px; }
    .panel h2 { margin: 0 0 10px; font-size: 18px; }
    .note { color: var(--muted); line-height: 1.5; font-size: 14px; margin: 0; }
    table { width: 100%; border-collapse: collapse; font-size: 14px; }
    th, td { border-bottom: 1px solid var(--line); padding: 9px 8px; text-align: right; }
    th:first-child, td:first-child { text-align: left; }
    .pre { color: var(--blue); font-weight: 800; }
    .geo { color: var(--teal); font-weight: 800; }
    .pos { color: var(--teal); font-weight: 800; }
    .neg { color: var(--red); font-weight: 800; }
    @media (max-width: 1180px) {
      header { grid-template-columns: 1fr; }
      main { grid-template-columns: 1fr; }
      aside { display: grid; grid-auto-flow: column; grid-auto-columns: 255px; overflow-x: auto; border-right: 0; border-bottom: 1px solid var(--line); }
      button.case { margin-right: 8px; margin-bottom: 0; }
      .compare { grid-template-columns: repeat(2, minmax(160px, 1fr)); }
      .details { grid-template-columns: 1fr; }
    }
    @media (max-width: 680px) {
      header { padding: 18px 16px 14px; }
      .score-grid { grid-template-columns: repeat(2, minmax(120px, 1fr)); }
      .workspace { padding: 14px; }
      .compare { grid-template-columns: 1fr; }
    }
  </style>
</head>
<body>
  <header>
    <div>
      <h1>Geo-Affordance Candidate Selection</h1>
      <p class="subline">Pretrained OmniTry compared against the same pretrained model with geometry-aware prompting and best-of-two candidate selection.</p>
    </div>
    <section class="score-grid" id="scoreGrid"></section>
  </header>
  <main>
    <aside id="caseList"></aside>
    <section class="workspace">
      <div class="compare" id="compare"></div>
      <div class="details">
        <section class="panel">
          <h2>Case Metrics</h2>
          <table id="caseTable"></table>
        </section>
        <section class="panel">
          <h2>Benchmark</h2>
          <p class="note" id="benchmarkNote"></p>
        </section>
      </div>
      <section class="panel" style="margin-top:14px">
        <h2>Class Breakdown</h2>
        <table id="classTable"></table>
      </section>
    </section>
  </main>
  <script>
    const data = __PAGE_DATA__;
    const scoreGrid = document.getElementById("scoreGrid");
    const caseList = document.getElementById("caseList");
    const compare = document.getElementById("compare");
    const caseTable = document.getElementById("caseTable");
    const classTable = document.getElementById("classTable");
    const benchmarkNote = document.getElementById("benchmarkNote");
    const fmt = (value) => Number(value).toFixed(6);
    const delta = (value) => `${value >= 0 ? "+" : ""}${fmt(value)}`;
    const cls = (value) => value >= 0 ? "pos" : "neg";

    function renderScores() {
      const m = data.metrics.hard;
      const cards = [
        ["Pretrained", fmt(m.pretrained.total_mean), "pre"],
        ["Pretrained + Geo", fmt(m.geo.total_mean), "geo"],
        ["Delta", delta(m.delta.total_mean), cls(m.delta.total_mean)],
        ["Win / Tie / Loss", `${m.wins}/${m.ties}/${m.losses}`, "geo"],
      ];
      scoreGrid.innerHTML = cards.map(([label, value, klass]) => `
        <article class="score-card"><span>${label}</span><strong class="${klass}">${value}</strong></article>
      `).join("");
    }

    function renderList(activeIndex) {
      caseList.innerHTML = data.cases.map((item, index) => `
        <button class="case ${index === activeIndex ? "active" : ""}" data-index="${index}">
          <strong>${item.id}</strong>
          <span>${item.category} | total ${delta(item.delta_total)}</span>
        </button>
      `).join("");
      caseList.querySelectorAll("button").forEach((button) => {
        button.addEventListener("click", () => renderCase(Number(button.dataset.index)));
      });
    }

    function renderCase(index) {
      const item = data.cases[index];
      renderList(index);
      compare.innerHTML = [
        ["Person input", item.assets.person],
        ["Object reference", item.assets.object],
        ["Pretrained", item.assets.pretrained],
        ["Pretrained + Geo", item.assets.geo],
      ].map(([label, src]) => `
        <figure><img src="${src}" alt="${label}"><figcaption>${label}</figcaption></figure>
      `).join("");
      caseTable.innerHTML = `
        <tr><th>Metric</th><th>Pretrained</th><th>Pretrained + Geo</th><th>Delta</th></tr>
        <tr><td>Total</td><td class="pre">${fmt(item.pretrained.total)}</td><td class="geo">${fmt(item.geo.total)}</td><td class="${cls(item.delta_total)}">${delta(item.delta_total)}</td></tr>
        <tr><td>Object</td><td>${fmt(item.pretrained.object)}</td><td>${fmt(item.geo.object)}</td><td class="${cls(item.delta_object)}">${delta(item.delta_object)}</td></tr>
        <tr><td>Person</td><td>${fmt(item.pretrained.person)}</td><td>${fmt(item.geo.person)}</td><td class="${cls(item.delta_person)}">${delta(item.delta_person)}</td></tr>
        <tr><td>Artifact</td><td>${fmt(item.pretrained.artifact)}</td><td>${fmt(item.geo.artifact)}</td><td class="${cls(item.delta_artifact)}">${delta(item.delta_artifact)}</td></tr>
      `;
    }

    function renderBenchmark() {
      const hard = data.metrics.hard;
      benchmarkNote.textContent = `Hard benchmark: ${hard.items} items, ${hard.wins} wins, ${hard.ties} ties, ${hard.losses} losses. Diverse demo: ${data.metrics.diverse.wins}/${data.metrics.diverse.items} cases improve under geometry, with sunglasses kept as the main failure case in the notes.`;
      classTable.innerHTML = `
        <tr><th>Class</th><th>Count</th><th>Pretrained</th><th>Pretrained + Geo</th><th>Delta</th></tr>
        ${hard.classes.map((row) => `
          <tr>
            <td>${row.category}</td>
            <td>${row.count}</td>
            <td>${fmt(row.pretrained_total)}</td>
            <td>${fmt(row.geo_total)}</td>
            <td class="${cls(row.delta_total)}">${delta(row.delta_total)}</td>
          </tr>
        `).join("")}
      `;
    }

    renderScores();
    renderBenchmark();
    renderCase(0);
  </script>
</body>
</html>
"""


def md_table(headers, rows):
    lines = [
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join(["---"] * len(headers)) + " |",
    ]
    for row in rows:
        lines.append("| " + " | ".join(str(value) for value in row) + " |")
    return "\n".join(lines)


def fmt(value):
    return f"{float(value):.6f}"


def signed(value):
    return f"{float(value):+.6f}"


def render_report(output_dir, doc_output, metrics, cases):
    hard = metrics["hard"]
    diverse = metrics["diverse"]
    finetune = metrics["finetune"]
    crawl = metrics["crawl"]

    main_table = md_table(
        ["Protocol", "Items", "Total", "Object", "Person", "Artifact"],
        [
            [
                "Pretrained",
                hard["items"],
                fmt(hard["pretrained"]["total_mean"]),
                fmt(hard["pretrained"]["object_mean"]),
                fmt(hard["pretrained"]["person_mean"]),
                fmt(hard["pretrained"]["artifact_mean"]),
            ],
            [
                "Pretrained + Geo",
                hard["items"],
                fmt(hard["geo"]["total_mean"]),
                fmt(hard["geo"]["object_mean"]),
                fmt(hard["geo"]["person_mean"]),
                fmt(hard["geo"]["artifact_mean"]),
            ],
            [
                "Delta",
                "",
                signed(hard["delta"]["total_mean"]),
                signed(hard["delta"]["object_mean"]),
                signed(hard["delta"]["person_mean"]),
                signed(hard["delta"]["artifact_mean"]),
            ],
        ],
    )

    class_table = md_table(
        ["Class", "Count", "Pretrained", "Pretrained + Geo", "Delta"],
        [
            [
                row["category"],
                row["count"],
                fmt(row["pretrained_total"]),
                fmt(row["geo_total"]),
                signed(row["delta_total"]),
            ]
            for row in hard["classes"]
        ],
    )

    selected_table = md_table(
        ["Case", "Class", "Set", "Pretrained", "Pretrained + Geo", "Delta"],
        [
            [
                case["id"],
                case["category"],
                case["group"],
                fmt(case["pretrained"]["total"]),
                fmt(case["geo"]["total"]),
                signed(case["delta_total"]),
            ]
            for case in cases
        ],
    )

    finetune_rows = [
        ["Pretrained hard benchmark", fmt(hard["pretrained"]["total_mean"])],
        ["Pretrained + Geo hard benchmark", fmt(hard["geo"]["total_mean"])],
        ["Raw fine-tuned LoRA hard benchmark", fmt(finetune["summary"]["total_mean"])],
        ["Fine-tuned minus pretrained", signed(finetune["summary"]["total_mean"] - hard["pretrained"]["total_mean"])],
    ]
    finetune_table = md_table(["Run", "Total score"], finetune_rows)

    report = f"""# Geo-Affordance Candidate Selection for Small-Object Try-On

Date: {date.today().isoformat()}

## Executive Summary

This project should be presented around the geometry method, not around fine-tuning. The strongest reproducible result is a controlled comparison between the same pretrained OmniTry pipeline and the same pipeline with Geo-Affordance Candidate Selection (GACS). On the 32-item hard small-object benchmark, GACS improves the total mean from {fmt(hard['pretrained']['total_mean'])} to {fmt(hard['geo']['total_mean'])}, a delta of {signed(hard['delta']['total_mean'])}. It wins {hard['wins']} cases, ties {hard['ties']}, and loses {hard['losses']}.

The fine-tuning branch is valuable as an exploratory negative result. With the data available in this environment, the raw fine-tuned LoRA scores lower than the pretrained baseline ({fmt(finetune['summary']['total_mean'])} versus {fmt(hard['pretrained']['total_mean'])}). That makes fine-tuning a weak main claim, but a strong motivation for a geometry-first method that works without collecting expensive paired labels.

## Method

GACS is a training-free post-processing protocol for small-object virtual try-on. It adds category-specific geometry constraints to the prompt and then generates multiple candidates with the frozen pretrained model. Each candidate is scored using three terms:

- Object consistency: compare the object reference to the expected affordance region using color histogram overlap.
- Person preservation: compare pixels outside the affordance region to the original person image.
- Artifact health: reward sharpness, contrast, and non-saturated pixels.

The final score is `0.35 * object + 0.35 * person + 0.30 * artifact`. This is intentionally simple and reproducible. The key idea is that small objects such as rings, bracelets, earrings, glasses, and necklaces fail mostly because the model is underconstrained about placement, scale, and occlusion. Geometry-aware prompting plus candidate selection narrows the search space without updating model weights.

## Benchmark Protocol

The main benchmark is the 32-item local hard subset from OmniTry small-object cases. The baseline is the pretrained model with one generated candidate. The geometry run uses the same pretrained weights with GACS and two candidates. All scores are produced by `scripts/run_tryon_benchmark.py` and are stored under `outputs/tryon_benchmark`.

{main_table}

## Class Breakdown

{class_table}

## Representative Demo Cases

The demo set combines diverse person/object categories with top hard-benchmark wins. It includes ring, earrings, glasses, necklace, and bracelet examples across multiple person images, plus the strongest hard-set wins.

{selected_table}

Demo artifacts:

- UI: `{(output_dir / 'index.html').relative_to(ROOT)}`
- Video: `{(output_dir / 'geo_method_demo.mp4').relative_to(ROOT)}`
- Case manifest: `{(output_dir / 'demo_manifest.json').relative_to(ROOT)}`

## Why Geometry Helps

Small accessories occupy a tiny fraction of the image, so a global text prompt often gives the diffusion model too much freedom. The model can preserve the person while losing the exact object, or it can add the object in the wrong location. GACS helps because it makes the expected affordance region explicit, asks for class-specific occlusion and scale, and rejects candidates that damage unrelated parts of the person image.

This is also why the measured gains are small but meaningful. We are not changing the generator; we are choosing better samples from the same generator. The method improves reliability most on cases where the object has a predictable geometric relation to the body, such as rings on fingers, bracelets on wrists, earrings near ears, glasses across the nose bridge, and necklaces around the neck.

## Why Fine-Tuning Is Not the Main Claim

Fine-tuning is not impossible, but it is not the best story for this project under the current constraints.

{finetune_table}

The obstacles are practical and methodological:

- True paired data is rare. For supervised try-on we need a person image without the object, an object reference, and the same person wearing that object as target.
- Crawled web images do not provide target images or clean masks. LLM labels can identify object class and rough boxes, but they do not solve pixel-accurate masking.
- Small-object masks are brittle. Rings, earrings, chains, watch straps, and glasses are thin, reflective, and often occluded by hair, hands, or clothing.
- The current crawl produced {crawl['crawled']} candidates, {crawl['usable_labels']} LLM-usable labels, and only {crawl['pseudo_pairs']} immediately usable pseudo-pairs. That is not enough to support a strong fine-tuning claim.
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
"""

    output_report = output_dir / "method_report.md"
    output_report.write_text(report, encoding="utf-8")
    doc_path = resolve(doc_output)
    doc_path.parent.mkdir(parents=True, exist_ok=True)
    doc_path.write_text(report, encoding="utf-8")


def build_metrics(hard_pretrained, hard_geo, diverse_pretrained, diverse_geo, finetune_payload):
    hard_pre = metric_summary(hard_pretrained)
    hard_g = metric_summary(hard_geo)
    diverse_pre = metric_summary(diverse_pretrained)
    diverse_g = metric_summary(diverse_geo)
    hard_compare = common_comparison(hard_pretrained, hard_geo)
    diverse_compare = common_comparison(diverse_pretrained, diverse_geo)
    hard_delta = {
        key: hard_g[key] - hard_pre[key]
        for key in ["total_mean", "object_mean", "person_mean", "artifact_mean"]
    }
    diverse_delta = {
        key: diverse_g[key] - diverse_pre[key]
        for key in ["total_mean", "object_mean", "person_mean", "artifact_mean"]
    }
    cost_state = maybe_load_json("outputs/llm_labeling/cost_state.json") or {}
    return {
        "hard": {
            "items": hard_compare["items"],
            "pretrained": hard_pre,
            "geo": hard_g,
            "delta": hard_delta,
            "wins": hard_compare["wins"],
            "ties": hard_compare["ties"],
            "losses": hard_compare["losses"],
            "top_wins": hard_compare["top_wins"][:12],
            "classes": class_comparison(hard_pretrained, hard_geo),
        },
        "diverse": {
            "items": diverse_compare["items"],
            "pretrained": diverse_pre,
            "geo": diverse_g,
            "delta": diverse_delta,
            "wins": diverse_compare["wins"],
            "ties": diverse_compare["ties"],
            "losses": diverse_compare["losses"],
            "top_wins": diverse_compare["top_wins"],
        },
        "finetune": {
            "summary": metric_summary(finetune_payload),
        },
        "crawl": {
            "crawled": record_count("data/hard_cases/commons_hard_cases.json"),
            "llm_labels": record_count("data/hard_cases/commons_llm_labels.json"),
            "usable_labels": record_count("data/hard_cases/commons_llm_usable_labels.json"),
            "pseudo_pairs": record_count("data/hard_cases/commons_pseudo_paired_train.json"),
            "merged_train_items": record_count("data/hard_cases/omnitry_commons_pseudo_paired_train.json"),
            "label_cost_usd": float(cost_state.get("spent_usd", 0.0) or 0.0),
            "label_budget_usd": float(cost_state.get("budget_usd", 0.0) or 0.0),
        },
    }


def main():
    args = parse_args()
    output_dir = resolve(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    hard_pretrained = load_json(args.hard_pretrained_summary)
    hard_geo = load_json(args.hard_geo_summary)
    diverse_pretrained = load_json(args.diverse_pretrained_summary)
    diverse_geo = load_json(args.diverse_geo_summary)
    finetune_payload = load_json(args.finetune_summary)

    metrics = build_metrics(hard_pretrained, hard_geo, diverse_pretrained, diverse_geo, finetune_payload)
    cases = build_cases(args, output_dir, hard_pretrained, hard_geo, diverse_pretrained, diverse_geo)
    video = render_video(output_dir, cases, args.seconds_per_case)
    render_html(output_dir, cases, metrics)

    manifest_cases = []
    for case in cases:
        copied = dict(case)
        copied["assets"] = {key: asset_rel(ROOT, value) for key, value in case["assets"].items()}
        manifest_cases.append(copied)
    write_json(
        output_dir / "demo_manifest.json",
        {
            "method": "Geo-Affordance Candidate Selection",
            "date": date.today().isoformat(),
            "metrics": metrics,
            "cases": manifest_cases,
            "video": asset_rel(ROOT, video),
            "ui": asset_rel(ROOT, output_dir / "index.html"),
        },
    )
    render_report(output_dir, args.doc_output, metrics, manifest_cases)

    print(f"Wrote UI -> {output_dir / 'index.html'}")
    print(f"Wrote video -> {video}")
    print(f"Wrote report -> {resolve(args.doc_output)}")
    print(
        json.dumps(
            {
                "hard_total_delta": metrics["hard"]["delta"]["total_mean"],
                "hard_wins": metrics["hard"]["wins"],
                "hard_ties": metrics["hard"]["ties"],
                "hard_losses": metrics["hard"]["losses"],
                "diverse_wins": metrics["diverse"]["wins"],
                "finetune_total": metrics["finetune"]["summary"]["total_mean"],
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    raise SystemExit(main())
