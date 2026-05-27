#!/usr/bin/env python
import argparse
import json
import shutil
import subprocess
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont, ImageOps


ROOT = Path(__file__).resolve().parents[1]


def parse_args():
    parser = argparse.ArgumentParser(description="Build a three-way geometry/dataset demo UI and video.")
    parser.add_argument("--manifest", default="data/hard_cases/omnitry_small_object_diverse_demo.json")
    parser.add_argument("--pretrained-summary", default="outputs/tryon_benchmark/diverse_demo_pretrained_summary.json")
    parser.add_argument("--gacs-summary", default="outputs/tryon_benchmark/diverse_demo_pretrained_gacs_summary.json")
    parser.add_argument("--small-data-summary", default="outputs/tryon_benchmark/diverse_demo_small_data_best_summary.json")
    parser.add_argument("--small-data-raw-summary", default="outputs/tryon_benchmark/diverse_demo_small_data_gacs_summary.json")
    parser.add_argument("--output-dir", default="outputs/demo/geo_affordance_three_way")
    parser.add_argument("--seconds-per-case", type=float, default=2.5)
    return parser.parse_args()


def load_json(path):
    with (ROOT / path).open("r", encoding="utf-8") as handle:
        return json.load(handle)


def write_json(path, payload):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2)
        handle.write("\n")


def resolve(path):
    path = Path(path)
    return path if path.is_absolute() else ROOT / path


def by_id(payload):
    return {row["id"]: row for row in payload.get("items", [])}


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


def copy_asset(src, dst):
    dst.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(src, dst)
    return dst


def safe_script_json(value):
    return str(value).replace("<", "\\u003c").replace(">", "\\u003e").replace("&", "\\u0026")


def font(name, size):
    for base in [Path("/usr/share/fonts/truetype/dejavu"), Path("/usr/share/fonts/dejavu")]:
        path = base / name
        if path.is_file():
            return ImageFont.truetype(str(path), size=size)
    return ImageFont.load_default()


def fit_image(path, size):
    image = ImageOps.exif_transpose(Image.open(path).convert("RGB"))
    canvas = Image.new("RGB", size, (250, 250, 250))
    fitted = ImageOps.contain(image, size)
    canvas.paste(fitted, ((size[0] - fitted.width) // 2, (size[1] - fitted.height) // 2))
    return canvas


def render_frame(case, output_path):
    width, height = 1920, 1080
    margin = 44
    gap = 20
    panel_w = 350
    panel_h = 690
    canvas = Image.new("RGB", (width, height), (247, 248, 251))
    draw = ImageDraw.Draw(canvas)
    title = font("DejaVuSans-Bold.ttf", 42)
    subtitle = font("DejaVuSans.ttf", 22)
    label = font("DejaVuSans-Bold.ttf", 18)
    score_font = font("DejaVuSans-Bold.ttf", 27)
    small = font("DejaVuSans.ttf", 20)
    ink = (17, 24, 39)
    muted = (75, 85, 99)
    blue = (37, 99, 235)
    teal = (15, 118, 110)
    amber = (180, 83, 9)

    draw.text((margin, 34), "Three-Way Geo-Affordance Demo", fill=ink, font=title)
    draw.text(
        (margin, 88),
        f"{case['id']} | {case['category']} | GACS {case['delta_gacs']:+.6f} | +data {case['delta_small_data']:+.6f}",
        fill=muted,
        font=subtitle,
    )
    x = 1130
    draw.text((x, 34), "PRETRAINED", fill=muted, font=label)
    draw.text((x, 68), f"{case['pretrained']['total']:.6f}", fill=blue, font=score_font)
    draw.text((x + 230, 34), "+ GACS", fill=muted, font=label)
    draw.text((x + 230, 68), f"{case['gacs']['total']:.6f}", fill=teal, font=score_font)
    draw.text((x + 440, 34), "+ DATA", fill=muted, font=label)
    draw.text((x + 440, 68), f"{case['small_data']['total']:.6f}", fill=amber, font=score_font)

    labels = [
        ("Person", case["assets"]["person"]),
        ("Object", case["assets"]["object"]),
        ("Pretrained", case["assets"]["pretrained"]),
        ("Pretrained + GACS", case["assets"]["gacs"]),
        ("GACS + Small Data", case["assets"]["small_data"]),
    ]
    top = 158
    for i, (name, path) in enumerate(labels):
        left = margin + i * (panel_w + gap)
        draw.rounded_rectangle((left, top, left + panel_w, top + panel_h), radius=8, fill=(255, 255, 255), outline=(209, 213, 219), width=2)
        draw.text((left + 16, top + 14), name, fill=ink, font=label)
        image = fit_image(path, (panel_w - 32, panel_h - 72))
        canvas.paste(image, (left + 16, top + 52))

    y = 886
    draw.text(
        (margin, y),
        f"object deltas: GACS {case['delta_gacs_object']:+.6f} | +data {case['delta_small_data_object']:+.6f}",
        fill=ink,
        font=small,
    )
    draw.text(
        (margin, y + 36),
        f"small-data source selected by GACS: {case.get('small_data_source', 'small_data_gacs')}",
        fill=muted,
        font=small,
    )
    draw.text(
        (margin, y + 72),
        "This is a reproducible demo benchmark on a diverse six-person small-object subset, not a paper-level full benchmark.",
        fill=muted,
        font=small,
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(output_path, quality=95)


def render_video(output_dir, cases, seconds_per_case):
    frames_dir = output_dir / "frames"
    frames_dir.mkdir(parents=True, exist_ok=True)
    for index, case in enumerate(cases):
        render_frame(case, frames_dir / f"frame_{index:03d}.jpg")
    video = output_dir / "three_way_demo.mp4"
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
    def rel(path):
        return Path(path).relative_to(output_dir).as_posix()

    cases_for_page = []
    for case in cases:
        copied = dict(case)
        copied["assets"] = {key: rel(value) for key, value in case["assets"].items()}
        cases_for_page.append(copied)

    page_json = json.dumps({"cases": cases_for_page, "metrics": metrics}, indent=2)
    html = f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>Three-Way Geo-Affordance Demo</title>
  <style>
    :root {{
      --bg: #f7f8fb;
      --panel: #ffffff;
      --ink: #111827;
      --muted: #4b5563;
      --line: #d1d5db;
      --blue: #2563eb;
      --teal: #0f766e;
      --amber: #b45309;
    }}
    * {{ box-sizing: border-box; }}
    body {{
      margin: 0;
      font-family: Inter, ui-sans-serif, system-ui, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
      background: var(--bg);
      color: var(--ink);
      letter-spacing: 0;
    }}
    header {{
      background: #fff;
      border-bottom: 1px solid var(--line);
      padding: 22px 26px 16px;
      display: grid;
      grid-template-columns: minmax(320px, 1fr) minmax(520px, .9fr);
      gap: 24px;
      align-items: end;
    }}
    h1 {{ margin: 0 0 6px; font-size: clamp(25px, 3vw, 38px); line-height: 1.08; }}
    .subline {{ margin: 0; color: var(--muted); font-size: 15px; max-width: 900px; }}
    .score-grid {{ display: grid; grid-template-columns: repeat(3, minmax(150px, 1fr)); gap: 8px; }}
    .score-card {{
      border: 1px solid var(--line);
      border-radius: 8px;
      background: #fff;
      padding: 11px 12px;
    }}
    .score-card span {{ display: block; color: var(--muted); font-size: 11px; font-weight: 800; text-transform: uppercase; }}
    .score-card strong {{ display: block; margin-top: 4px; font-size: 20px; }}
    main {{ display: grid; grid-template-columns: 310px minmax(0, 1fr); min-height: calc(100vh - 112px); }}
    aside {{ background: #fff; border-right: 1px solid var(--line); padding: 14px; overflow: auto; }}
    button.case {{
      width: 100%;
      min-height: 78px;
      border: 1px solid var(--line);
      border-radius: 8px;
      background: #fff;
      text-align: left;
      padding: 12px;
      margin-bottom: 8px;
      cursor: pointer;
      color: var(--ink);
    }}
    button.case.active {{ border-color: var(--teal); box-shadow: 0 0 0 2px rgba(15,118,110,.14); }}
    button.case strong {{ display: block; font-size: 14px; overflow-wrap: anywhere; }}
    button.case span {{ display: block; margin-top: 6px; color: var(--muted); font-size: 13px; }}
    .workspace {{ padding: 18px 22px 28px; overflow: auto; }}
    .compare {{ display: grid; grid-template-columns: repeat(5, minmax(150px, 1fr)); gap: 12px; }}
    figure {{ margin: 0; border: 1px solid var(--line); border-radius: 8px; background: var(--panel); overflow: hidden; }}
    figure img {{ width: 100%; aspect-ratio: 4 / 5; object-fit: contain; background: #fafafa; display: block; }}
    figcaption {{ border-top: 1px solid var(--line); padding: 10px 11px; font-size: 13px; font-weight: 800; }}
    .details {{ display: grid; grid-template-columns: 1.25fr .75fr; gap: 14px; margin-top: 14px; align-items: start; }}
    .panel {{ border: 1px solid var(--line); border-radius: 8px; background: #fff; padding: 14px; }}
    .panel h2 {{ margin: 0 0 10px; font-size: 18px; }}
    table {{ width: 100%; border-collapse: collapse; font-size: 14px; }}
    th, td {{ border-bottom: 1px solid var(--line); padding: 9px 8px; text-align: right; }}
    th:first-child, td:first-child {{ text-align: left; }}
    .pre {{ color: var(--blue); font-weight: 800; }}
    .gacs {{ color: var(--teal); font-weight: 800; }}
    .data {{ color: var(--amber); font-weight: 800; }}
    .pos {{ color: var(--teal); font-weight: 800; }}
    .neg {{ color: #b91c1c; font-weight: 800; }}
    .note {{ color: var(--muted); line-height: 1.5; font-size: 14px; margin: 0; }}
    @media (max-width: 1250px) {{
      header {{ grid-template-columns: 1fr; }}
      main {{ grid-template-columns: 1fr; }}
      aside {{ display: grid; grid-auto-flow: column; grid-auto-columns: 250px; overflow-x: auto; border-right: 0; border-bottom: 1px solid var(--line); }}
      button.case {{ margin-right: 8px; margin-bottom: 0; }}
      .compare {{ grid-template-columns: repeat(2, minmax(160px, 1fr)); }}
      .details {{ grid-template-columns: 1fr; }}
    }}
    @media (max-width: 640px) {{
      header {{ padding: 18px 15px 12px; }}
      .workspace {{ padding: 14px; }}
      .score-grid {{ grid-template-columns: 1fr; }}
      .compare {{ grid-template-columns: 1fr; }}
    }}
  </style>
</head>
<body>
  <header>
    <div>
      <h1>Three-Way Geo-Affordance Demo</h1>
      <p class="subline">Same diverse-person small-object subset, comparing pretrained, pretrained with geometry candidate selection, and geometry selection with the small-object pseudo-pair adapter in the candidate pool.</p>
    </div>
    <section class="score-grid">
      <div class="score-card"><span>pretrained</span><strong id="m-pre"></strong></div>
      <div class="score-card"><span>pretrained + GACS</span><strong id="m-gacs"></strong></div>
      <div class="score-card"><span>GACS + data</span><strong id="m-data"></strong></div>
    </section>
  </header>
  <main>
    <aside id="case-list"></aside>
    <section class="workspace">
      <section class="compare">
        <figure><img id="person" alt="person input"><figcaption>Person</figcaption></figure>
        <figure><img id="object" alt="object reference"><figcaption>Object</figcaption></figure>
        <figure><img id="pretrained" alt="pretrained output"><figcaption>Pretrained</figcaption></figure>
        <figure><img id="gacs" alt="GACS output"><figcaption>Pretrained + GACS</figcaption></figure>
        <figure><img id="small-data" alt="small data output"><figcaption>GACS + Small Data</figcaption></figure>
      </section>
      <section class="details">
        <div class="panel">
          <h2 id="case-title"></h2>
          <table>
            <thead><tr><th>metric</th><th>pretrained</th><th>+GACS</th><th>+data</th><th>GACS delta</th><th>data delta</th></tr></thead>
            <tbody id="score-body"></tbody>
          </table>
        </div>
        <div class="panel">
          <h2>Method</h2>
          <p class="note">GACS scores candidates by object preservation in the affordance crop, person preservation outside it, and artifact health. The data-augmented row uses the same selector over the pretrained candidate and the small-object pseudo-pair fine-tuned candidate, so weak fine-tuned cases can be rejected instead of hurting the final output.</p>
        </div>
      </section>
    </section>
  </main>
  <script type="application/json" id="page-data">{safe_script_json(page_json)}</script>
  <script>
    const page = JSON.parse(document.getElementById('page-data').textContent);
    const cases = page.cases;
    const metrics = page.metrics;
    const fmt = value => Number(value).toFixed(6);
    const signed = value => (value >= 0 ? '+' : '') + Number(value).toFixed(6);
    document.getElementById('m-pre').textContent = fmt(metrics.pretrained.total_mean);
    document.getElementById('m-gacs').textContent = fmt(metrics.gacs.total_mean) + ' (' + signed(metrics.gacs.total_mean - metrics.pretrained.total_mean) + ')';
    document.getElementById('m-data').textContent = fmt(metrics.small_data.total_mean) + ' (' + signed(metrics.small_data.total_mean - metrics.pretrained.total_mean) + ')';

    function cls(v) {{ return v >= 0 ? 'pos' : 'neg'; }}
    function row(name, item, key, deltaG, deltaD) {{
      return `<tr><td>${{name}}</td><td class="pre">${{fmt(item.pretrained[key])}}</td><td class="gacs">${{fmt(item.gacs[key])}}</td><td class="data">${{fmt(item.small_data[key])}}</td><td class="${{cls(deltaG)}}">${{signed(deltaG)}}</td><td class="${{cls(deltaD)}}">${{signed(deltaD)}}</td></tr>`;
    }}
    function selectCase(index) {{
      const item = cases[index];
      document.querySelectorAll('button.case').forEach((button, i) => button.classList.toggle('active', i === index));
      document.getElementById('person').src = item.assets.person;
      document.getElementById('object').src = item.assets.object;
      document.getElementById('pretrained').src = item.assets.pretrained;
      document.getElementById('gacs').src = item.assets.gacs;
      document.getElementById('small-data').src = item.assets.small_data;
      document.getElementById('case-title').textContent = `${{item.id}} (${{item.category}})`;
      document.getElementById('score-body').innerHTML = [
        row('total', item, 'total', item.delta_gacs, item.delta_small_data),
        row('object', item, 'object', item.delta_gacs_object, item.delta_small_data_object),
        row('person', item, 'person', item.delta_gacs_person, item.delta_small_data_person),
        row('artifact', item, 'artifact', item.delta_gacs_artifact, item.delta_small_data_artifact)
      ].join('');
    }}
    cases.forEach((item, index) => {{
      const button = document.createElement('button');
      button.className = 'case';
      button.innerHTML = `<strong>${{item.id}}</strong><span>${{item.category}} | GACS ${{signed(item.delta_gacs)}} | data ${{signed(item.delta_small_data)}}</span>`;
      button.addEventListener('click', () => selectCase(index));
      document.getElementById('case-list').appendChild(button);
    }});
    selectCase(0);
  </script>
</body>
</html>
"""
    (output_dir / "index.html").write_text(html, encoding="utf-8")


def render_method(output_dir, metrics, cases, raw_metrics, small_data_sources):
    lines = [
        "# Three-Way Geo-Affordance Demo",
        "",
        "This is a reproducible demo benchmark on a diverse six-person small-object subset. It is not a full paper benchmark.",
        "",
        "## Benchmark Table",
        "",
        "| system | total | object | person | artifact | delta total vs pretrained |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    base = metrics["pretrained"]
    for name, key in [
        ("Pretrained", "pretrained"),
        ("Pretrained + GACS", "gacs"),
        ("GACS + small-object data", "small_data"),
    ]:
        row = metrics[key]
        lines.append(
            f"| {name} | {row['total_mean']:.6f} | {row['object_mean']:.6f} | "
            f"{row['person_mean']:.6f} | {row['artifact_mean']:.6f} | "
            f"{row['total_mean'] - base['total_mean']:+.6f} |"
        )
    if raw_metrics:
        lines.extend(
            [
                "",
                "Raw fine-tuned + GACS before candidate-bank selection:",
                "",
                f"- total mean: `{raw_metrics['total_mean']:.6f}`",
                f"- delta vs pretrained: `{raw_metrics['total_mean'] - base['total_mean']:+.6f}`",
            ]
        )
    lines.extend(
        [
            "",
            f"Small-data candidate sources selected: `{small_data_sources}`",
            "",
            "## Cases",
            "",
            "| id | category | pretrained | +GACS | +data | data source |",
            "|---|---|---:|---:|---:|---|",
        ]
    )
    for case in cases:
        lines.append(
            f"| {case['id']} | {case['category']} | {case['pretrained']['total']:.6f} | "
            f"{case['gacs']['total']:.6f} | {case['small_data']['total']:.6f} | "
            f"{case.get('small_data_source', '')} |"
        )
    (output_dir / "method.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main():
    args = parse_args()
    output_dir = ROOT / args.output_dir
    assets_dir = output_dir / "assets"
    output_dir.mkdir(parents=True, exist_ok=True)
    assets_dir.mkdir(parents=True, exist_ok=True)

    manifest = load_json(args.manifest)
    manifest_by_id = {item["id"]: item for item in manifest.get("items", manifest)}
    pretrained_payload = load_json(args.pretrained_summary)
    gacs_payload = load_json(args.gacs_summary)
    small_data_payload = load_json(args.small_data_summary)
    raw_payload = load_json(args.small_data_raw_summary) if (ROOT / args.small_data_raw_summary).is_file() else None

    pretrained = by_id(pretrained_payload)
    gacs = by_id(gacs_payload)
    small_data = by_id(small_data_payload)
    ids = [item["id"] for item in manifest.get("items", []) if item["id"] in pretrained and item["id"] in gacs and item["id"] in small_data]
    cases = []
    for item_id in ids:
        item = manifest_by_id[item_id]
        rows = {
            "pretrained": pretrained[item_id],
            "gacs": gacs[item_id],
            "small_data": small_data[item_id],
        }
        assets = {
            "person": assets_dir / f"{item_id}__person.jpg",
            "object": assets_dir / f"{item_id}__object.jpg",
            "pretrained": assets_dir / f"{item_id}__pretrained.jpg",
            "gacs": assets_dir / f"{item_id}__gacs.jpg",
            "small_data": assets_dir / f"{item_id}__small_data.jpg",
        }
        copy_asset(resolve(item["person_path"]), assets["person"])
        copy_asset(resolve(item["object_path"]), assets["object"])
        copy_asset(resolve(rows["pretrained"]["image"]), assets["pretrained"])
        copy_asset(resolve(rows["gacs"]["image"]), assets["gacs"])
        copy_asset(resolve(rows["small_data"]["image"]), assets["small_data"])
        case = {
            "id": item_id,
            "category": item.get("category", rows["pretrained"].get("category", "")),
            "pretrained": {key: rows["pretrained"][key] for key in ["total", "object", "person", "artifact"]},
            "gacs": {key: rows["gacs"][key] for key in ["total", "object", "person", "artifact"]},
            "small_data": {key: rows["small_data"][key] for key in ["total", "object", "person", "artifact"]},
            "small_data_source": rows["small_data"].get("source", "small_data_gacs"),
            "assets": assets,
        }
        for suffix, key in [("", "total"), ("_object", "object"), ("_person", "person"), ("_artifact", "artifact")]:
            case[f"delta_gacs{suffix}"] = round(case["gacs"][key] - case["pretrained"][key], 6)
            case[f"delta_small_data{suffix}"] = round(case["small_data"][key] - case["pretrained"][key], 6)
        cases.append(case)

    metrics = {
        "pretrained": metric_summary(pretrained_payload),
        "gacs": metric_summary(gacs_payload),
        "small_data": metric_summary(small_data_payload),
        "small_data_raw": metric_summary(raw_payload) if raw_payload else None,
        "small_data_sources": small_data_payload.get("source_counts", {}),
    }
    serializable_cases = []
    for case in cases:
        copied = dict(case)
        copied["assets"] = {key: str(value) for key, value in case["assets"].items()}
        serializable_cases.append(copied)
    write_json(
        output_dir / "demo_manifest.json",
        {
            "method": "Geo-Affordance Candidate Selection",
            "source_summaries": {
                "pretrained": args.pretrained_summary,
                "gacs": args.gacs_summary,
                "small_data": args.small_data_summary,
                "small_data_raw": args.small_data_raw_summary,
            },
            "metrics": metrics,
            "cases": serializable_cases,
        },
    )
    render_html(output_dir, cases, metrics)
    render_method(output_dir, metrics, cases, metrics.get("small_data_raw"), metrics.get("small_data_sources", {}))
    video = render_video(output_dir, cases, args.seconds_per_case)
    print(json.dumps(metrics, indent=2))
    print(f"Wrote UI -> {output_dir / 'index.html'}")
    print(f"Wrote video -> {video}")
    print(f"Wrote manifest -> {output_dir / 'demo_manifest.json'}")


if __name__ == "__main__":
    raise SystemExit(main())
