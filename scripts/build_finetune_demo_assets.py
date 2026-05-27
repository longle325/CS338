#!/usr/bin/env python
import argparse
import json
import shutil
import subprocess
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont, ImageOps


ROOT = Path(__file__).resolve().parents[1]


def parse_args():
    parser = argparse.ArgumentParser(description="Build UI/video assets for fine-tune winner comparisons.")
    parser.add_argument("--manifest", default="data/hard_cases/omnitry_full_local_hard_cases.json")
    parser.add_argument("--original-summary", default="outputs/tryon_benchmark/original_enhanced_summary.json")
    parser.add_argument("--finetuned-summary", default="outputs/tryon_benchmark/enhanced_ft_summary.json")
    parser.add_argument("--best-summary", default="outputs/tryon_benchmark/best_total_3way_summary.json")
    parser.add_argument("--output-dir", default="outputs/demo/geo_affordance_finetune_wins")
    parser.add_argument("--max-cases", type=int, default=8)
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


def rows_by_id(payload):
    return {row["id"]: row for row in payload.get("items", [])}


def resolve(path):
    path = Path(path)
    if path.is_absolute():
        return path
    return ROOT / path


def summary_metrics(payload):
    summary = payload.get("summary", {})
    return {
        "items": summary.get("items", 0),
        "total_mean": float(summary.get("total_mean", 0.0)),
        "object_mean": float(summary.get("object_mean", 0.0)),
        "person_mean": float(summary.get("person_mean", 0.0)),
        "artifact_mean": float(summary.get("artifact_mean", 0.0)),
    }


def copy_image(src, dst):
    dst.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(src, dst)
    return dst


def safe_script_json(value):
    return str(value).replace("<", "\\u003c").replace(">", "\\u003e").replace("&", "\\u0026")


def find_font(name, size):
    candidates = [
        Path("/usr/share/fonts/truetype/dejavu") / name,
        Path("/usr/share/fonts/dejavu") / name,
    ]
    for path in candidates:
        if path.is_file():
            return ImageFont.truetype(str(path), size=size)
    return ImageFont.load_default()


def fit_image(image, box_size, background=(247, 248, 251)):
    image = ImageOps.exif_transpose(image.convert("RGB"))
    canvas = Image.new("RGB", box_size, background)
    fitted = ImageOps.contain(image, box_size)
    x = (box_size[0] - fitted.width) // 2
    y = (box_size[1] - fitted.height) // 2
    canvas.paste(fitted, (x, y))
    return canvas


def draw_label(draw, xy, label, value, font_label, font_value, accent):
    x, y = xy
    draw.text((x, y), label.upper(), fill=(77, 85, 97), font=font_label)
    draw.text((x, y + 30), value, fill=accent, font=font_value)


def render_frame(case, frame_path):
    width, height = 1920, 1080
    margin = 56
    canvas = Image.new("RGB", (width, height), (247, 248, 251))
    draw = ImageDraw.Draw(canvas)

    title_font = find_font("DejaVuSans-Bold.ttf", 44)
    subtitle_font = find_font("DejaVuSans.ttf", 24)
    label_font = find_font("DejaVuSans-Bold.ttf", 18)
    value_font = find_font("DejaVuSans-Bold.ttf", 30)
    small_font = find_font("DejaVuSans.ttf", 20)
    header_color = (17, 24, 39)
    accent = (15, 118, 110)
    muted = (75, 85, 99)

    draw.text((margin, 38), "Geo-Affordance Fine-Tune Wins", fill=header_color, font=title_font)
    subtitle = f"{case['id']} | {case['category']} | total +{case['delta_total']:.6f}"
    draw.text((margin, 94), subtitle, fill=muted, font=subtitle_font)

    score_x = 1210
    draw_label(draw, (score_x, 42), "pretrained", f"{case['original']['total']:.6f}", label_font, value_font, (37, 99, 235))
    draw_label(draw, (score_x + 230, 42), "fine-tuned", f"{case['finetuned']['total']:.6f}", label_font, value_font, accent)
    draw_label(draw, (score_x + 455, 42), "delta", f"+{case['delta_total']:.6f}", label_font, value_font, (180, 83, 9))

    panel_top = 160
    panel_w = 430
    panel_h = 730
    gap = 24
    labels = [
        ("Person input", case["assets"]["person"]),
        ("Object reference", case["assets"]["object"]),
        ("Pretrained", case["assets"]["original"]),
        ("Fine-tuned", case["assets"]["finetuned"]),
    ]
    for idx, (label, src) in enumerate(labels):
        x = margin + idx * (panel_w + gap)
        y = panel_top
        draw.rounded_rectangle((x, y, x + panel_w, y + panel_h), radius=8, fill=(255, 255, 255), outline=(209, 213, 219), width=2)
        draw.text((x + 18, y + 16), label, fill=header_color, font=label_font)
        image = Image.open(src)
        fitted = fit_image(image, (panel_w - 36, panel_h - 74), background=(250, 250, 250))
        canvas.paste(fitted, (x + 18, y + 54))

    footer_y = 922
    detail = (
        f"object {case['delta_object']:+.6f}   "
        f"person {case['delta_person']:+.6f}   "
        f"artifact {case['delta_artifact']:+.6f}"
    )
    draw.text((margin, footer_y), detail, fill=header_color, font=small_font)
    draw.text(
        (margin, footer_y + 38),
        "Selection is based on the implemented benchmark score: object preservation, person preservation, and artifact health.",
        fill=muted,
        font=small_font,
    )
    frame_path.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(frame_path, quality=95)


def render_html(output_dir, cases, metrics):
    def rel(path):
        return Path(path).relative_to(output_dir).as_posix()

    data_cases = []
    for case in cases:
        copied = dict(case)
        copied["assets"] = {key: rel(Path(value)) for key, value in case["assets"].items()}
        data_cases.append(copied)

    page_data = json.dumps({"cases": data_cases, "metrics": metrics}, indent=2)
    html_text = f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>Geo-Affordance Fine-Tune Demo</title>
  <style>
    :root {{
      color-scheme: light;
      --bg: #f7f8fb;
      --panel: #ffffff;
      --ink: #111827;
      --muted: #4b5563;
      --line: #d1d5db;
      --teal: #0f766e;
      --blue: #2563eb;
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
      display: grid;
      grid-template-columns: minmax(260px, 1fr) auto;
      gap: 24px;
      align-items: end;
      padding: 24px 28px 16px;
      border-bottom: 1px solid var(--line);
      background: #fff;
    }}
    h1 {{
      margin: 0 0 6px;
      font-size: clamp(24px, 3vw, 38px);
      line-height: 1.08;
    }}
    .subline {{ margin: 0; color: var(--muted); font-size: 15px; max-width: 920px; }}
    .metrics {{
      display: grid;
      grid-template-columns: repeat(4, minmax(112px, 1fr));
      gap: 8px;
      min-width: 520px;
    }}
    .metric {{
      border: 1px solid var(--line);
      border-radius: 8px;
      background: #fff;
      padding: 10px 12px;
    }}
    .metric span {{ display: block; color: var(--muted); font-size: 11px; text-transform: uppercase; font-weight: 700; }}
    .metric strong {{ display: block; margin-top: 4px; font-size: 19px; }}
    main {{
      display: grid;
      grid-template-columns: 320px minmax(0, 1fr);
      min-height: calc(100vh - 110px);
    }}
    aside {{
      border-right: 1px solid var(--line);
      padding: 14px;
      overflow: auto;
      background: #fff;
    }}
    button.case {{
      width: 100%;
      border: 1px solid var(--line);
      border-radius: 8px;
      background: #fff;
      color: var(--ink);
      padding: 12px;
      margin-bottom: 8px;
      text-align: left;
      cursor: pointer;
      min-height: 76px;
    }}
    button.case.active {{ border-color: var(--teal); box-shadow: 0 0 0 2px rgba(15,118,110,.14); }}
    button.case strong {{ display: block; font-size: 14px; overflow-wrap: anywhere; }}
    button.case span {{ display: block; margin-top: 6px; color: var(--muted); font-size: 13px; }}
    .workspace {{ padding: 18px 22px 28px; overflow: auto; }}
    .compare {{
      display: grid;
      grid-template-columns: repeat(4, minmax(180px, 1fr));
      gap: 14px;
      align-items: start;
    }}
    figure {{
      margin: 0;
      border: 1px solid var(--line);
      border-radius: 8px;
      background: var(--panel);
      overflow: hidden;
    }}
    figure img {{
      width: 100%;
      aspect-ratio: 4 / 5;
      object-fit: contain;
      background: #fafafa;
      display: block;
    }}
    figcaption {{
      padding: 10px 12px;
      font-size: 13px;
      font-weight: 700;
      border-top: 1px solid var(--line);
    }}
    .details {{
      display: grid;
      grid-template-columns: 1.1fr .9fr;
      gap: 14px;
      margin-top: 14px;
      align-items: start;
    }}
    .panel {{
      border: 1px solid var(--line);
      border-radius: 8px;
      background: #fff;
      padding: 14px;
    }}
    .panel h2 {{ margin: 0 0 10px; font-size: 18px; }}
    table {{ width: 100%; border-collapse: collapse; font-size: 14px; }}
    th, td {{ padding: 9px 8px; border-bottom: 1px solid var(--line); text-align: right; }}
    th:first-child, td:first-child {{ text-align: left; }}
    .pos {{ color: var(--teal); font-weight: 800; }}
    .pre {{ color: var(--blue); font-weight: 800; }}
    .ft {{ color: var(--teal); font-weight: 800; }}
    .note {{ color: var(--muted); line-height: 1.5; font-size: 14px; margin: 0; }}
    .thumbs {{
      display: grid;
      grid-template-columns: repeat(auto-fill, minmax(170px, 1fr));
      gap: 10px;
      margin-top: 14px;
    }}
    .thumb {{
      border: 1px solid var(--line);
      border-radius: 8px;
      overflow: hidden;
      background: #fff;
      cursor: pointer;
    }}
    .thumb img {{ width: 100%; aspect-ratio: 1 / 1; object-fit: cover; display: block; }}
    .thumb span {{ display: block; padding: 8px; font-size: 12px; color: var(--muted); overflow-wrap: anywhere; }}
    @media (max-width: 1100px) {{
      header {{ grid-template-columns: 1fr; }}
      .metrics {{ min-width: 0; grid-template-columns: repeat(2, minmax(112px, 1fr)); }}
      main {{ grid-template-columns: 1fr; }}
      aside {{ display: grid; grid-auto-flow: column; grid-auto-columns: 260px; overflow-x: auto; border-right: 0; border-bottom: 1px solid var(--line); }}
      button.case {{ margin-right: 8px; margin-bottom: 0; }}
      .compare {{ grid-template-columns: repeat(2, minmax(150px, 1fr)); }}
      .details {{ grid-template-columns: 1fr; }}
    }}
    @media (max-width: 620px) {{
      header {{ padding: 18px 16px 12px; }}
      .workspace {{ padding: 14px; }}
      .compare {{ grid-template-columns: 1fr; }}
      .metrics {{ grid-template-columns: 1fr 1fr; }}
    }}
  </style>
</head>
<body>
  <header>
    <div>
      <h1>Geo-Affordance Fine-Tune Demo</h1>
      <p class="subline">Fine-tuned winner cases from the actual OmniTry benchmark outputs, with the geometry-aware selection result summarized beside the raw checkpoint comparison.</p>
    </div>
    <section class="metrics" aria-label="benchmark metrics">
      <div class="metric"><span>pretrained</span><strong id="m-original"></strong></div>
      <div class="metric"><span>fine-tuned</span><strong id="m-finetuned"></strong></div>
      <div class="metric"><span>reranked best</span><strong id="m-best"></strong></div>
      <div class="metric"><span>best delta</span><strong id="m-delta"></strong></div>
    </section>
  </header>
  <main>
    <aside id="case-list"></aside>
    <section class="workspace">
      <section class="compare">
        <figure><img id="person-img" alt="person input"><figcaption>Person Input</figcaption></figure>
        <figure><img id="object-img" alt="object reference"><figcaption>Object Reference</figcaption></figure>
        <figure><img id="original-img" alt="pretrained output"><figcaption>Pretrained Output</figcaption></figure>
        <figure><img id="finetuned-img" alt="fine-tuned output"><figcaption>Fine-Tuned Output</figcaption></figure>
      </section>
      <section class="details">
        <div class="panel">
          <h2 id="case-title"></h2>
          <table>
            <thead><tr><th>metric</th><th>pretrained</th><th>fine-tuned</th><th>delta</th></tr></thead>
            <tbody id="score-body"></tbody>
          </table>
        </div>
        <div class="panel">
          <h2>Method</h2>
          <p class="note">Geo-Affordance Candidate Selection scores generated candidates in the predicted placement region and outside it: object color/detail preservation inside the affordance crop, person preservation outside the crop, and image health to penalize artifacts. The enhanced pseudo-paired data teaches the adapter the same erased-region reconstruction geometry, while the final inference trick keeps the candidate whose geometry score is strongest.</p>
        </div>
      </section>
      <section class="thumbs" id="thumbs"></section>
    </section>
  </main>
  <script type="application/json" id="page-data">{safe_script_json(page_data)}</script>
  <script>
    const page = JSON.parse(document.getElementById('page-data').textContent);
    const cases = page.cases;
    const metrics = page.metrics;
    const fmt = value => Number(value).toFixed(6);
    const signed = value => (value >= 0 ? '+' : '') + Number(value).toFixed(6);

    document.getElementById('m-original').textContent = fmt(metrics.original.total_mean);
    document.getElementById('m-finetuned').textContent = fmt(metrics.finetuned.total_mean);
    document.getElementById('m-best').textContent = fmt(metrics.best.total_mean);
    document.getElementById('m-delta').textContent = signed(metrics.best.total_mean - metrics.original.total_mean);

    const caseList = document.getElementById('case-list');
    const thumbs = document.getElementById('thumbs');

    function scoreRow(name, original, finetuned, delta) {{
      const cls = delta >= 0 ? 'pos' : '';
      return `<tr><td>${{name}}</td><td class="pre">${{fmt(original)}}</td><td class="ft">${{fmt(finetuned)}}</td><td class="${{cls}}">${{signed(delta)}}</td></tr>`;
    }}

    function selectCase(index) {{
      const item = cases[index];
      document.querySelectorAll('button.case').forEach((button, i) => button.classList.toggle('active', i === index));
      document.getElementById('person-img').src = item.assets.person;
      document.getElementById('object-img').src = item.assets.object;
      document.getElementById('original-img').src = item.assets.original;
      document.getElementById('finetuned-img').src = item.assets.finetuned;
      document.getElementById('case-title').textContent = `${{item.id}} (${{item.category}})`;
      document.getElementById('score-body').innerHTML = [
        scoreRow('total', item.original.total, item.finetuned.total, item.delta_total),
        scoreRow('object', item.original.object, item.finetuned.object, item.delta_object),
        scoreRow('person', item.original.person, item.finetuned.person, item.delta_person),
        scoreRow('artifact', item.original.artifact, item.finetuned.artifact, item.delta_artifact)
      ].join('');
    }}

    cases.forEach((item, index) => {{
      const button = document.createElement('button');
      button.className = 'case';
      button.innerHTML = `<strong>${{item.id}}</strong><span>${{item.category}} | total ${{signed(item.delta_total)}}</span>`;
      button.addEventListener('click', () => selectCase(index));
      caseList.appendChild(button);

      const thumb = document.createElement('button');
      thumb.className = 'thumb';
      thumb.innerHTML = `<img alt="${{item.id}} fine-tuned output" src="${{item.assets.finetuned}}"><span>${{item.id}} ${{signed(item.delta_total)}}</span>`;
      thumb.addEventListener('click', () => selectCase(index));
      thumbs.appendChild(thumb);
    }});
    selectCase(0);
  </script>
</body>
</html>
"""
    (output_dir / "index.html").write_text(html_text, encoding="utf-8")


def render_method_doc(output_dir, cases, metrics):
    lines = [
        "# Geo-Affordance Candidate Selection",
        "",
        "This demo uses actual generated benchmark outputs. The base fine-tuned checkpoint is not claimed to beat the pretrained checkpoint on the full 32-item mean; the improvement comes from geometry-aware candidate selection and the hybrid best-of summary.",
        "",
        "## Current Scores",
        "",
        f"- Pretrained total mean: `{metrics['original']['total_mean']:.6f}`",
        f"- Fine-tuned total mean: `{metrics['finetuned']['total_mean']:.6f}`",
        f"- Geometry-reranked best total mean: `{metrics['best']['total_mean']:.6f}`",
        f"- Reranked delta vs pretrained: `{metrics['best']['total_mean'] - metrics['original']['total_mean']:+.6f}`",
        "",
        "## Method",
        "",
        "Geo-Affordance Candidate Selection (GACS) couples the enhanced pseudo-paired dataset with an inference-time selector. The pseudo pairs erase the object region and reconstruct the original target, so the adapter sees a localized geometry task instead of a free-form redraw. At inference, each candidate is scored by object preservation inside the class affordance crop, person preservation outside the crop, and artifact health. The final output keeps the candidate with the best combined geometry score.",
        "",
        "## Fine-Tuned Winner Cases",
        "",
        "| id | category | pretrained | fine-tuned | delta |",
        "|---|---|---:|---:|---:|",
    ]
    for case in cases:
        lines.append(
            f"| {case['id']} | {case['category']} | {case['original']['total']:.6f} | "
            f"{case['finetuned']['total']:.6f} | {case['delta_total']:+.6f} |"
        )
    lines.append("")
    (output_dir / "method.md").write_text("\n".join(lines), encoding="utf-8")


def render_video(output_dir, cases, seconds_per_case):
    frames_dir = output_dir / "frames"
    frames_dir.mkdir(parents=True, exist_ok=True)
    frame_paths = []
    for index, case in enumerate(cases):
        frame_path = frames_dir / f"frame_{index:03d}.jpg"
        render_frame(case, frame_path)
        frame_paths.append(frame_path)

    video_path = output_dir / "finetune_wins.mp4"
    framerate = 1.0 / max(seconds_per_case, 0.5)
    cmd = [
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
        str(video_path),
    ]
    subprocess.run(cmd, check=True)
    return video_path, frame_paths


def main():
    args = parse_args()
    output_dir = ROOT / args.output_dir
    assets_dir = output_dir / "assets"
    output_dir.mkdir(parents=True, exist_ok=True)
    assets_dir.mkdir(parents=True, exist_ok=True)

    manifest = load_json(args.manifest)
    manifest_items = manifest.get("items", manifest)
    manifest_by_id = {item["id"]: item for item in manifest_items}

    original_payload = load_json(args.original_summary)
    finetuned_payload = load_json(args.finetuned_summary)
    best_payload = load_json(args.best_summary)
    original = rows_by_id(original_payload)
    finetuned = rows_by_id(finetuned_payload)

    winners = []
    for item_id, ft_row in finetuned.items():
        orig_row = original.get(item_id)
        manifest_row = manifest_by_id.get(item_id)
        if not orig_row or not manifest_row:
            continue
        delta = ft_row["total"] - orig_row["total"]
        if delta <= 0:
            continue
        winners.append((delta, item_id, orig_row, ft_row, manifest_row))

    winners.sort(reverse=True, key=lambda row: row[0])
    winners = winners[: args.max_cases]
    if not winners:
        raise RuntimeError("No fine-tuned winner cases found.")

    cases = []
    for delta, item_id, orig_row, ft_row, manifest_row in winners:
        case_assets = {
            "person": assets_dir / f"{item_id}__person.jpg",
            "object": assets_dir / f"{item_id}__object.jpg",
            "original": assets_dir / f"{item_id}__pretrained.jpg",
            "finetuned": assets_dir / f"{item_id}__finetuned.jpg",
        }
        copy_image(resolve(manifest_row["person_path"]), case_assets["person"])
        copy_image(resolve(manifest_row["object_path"]), case_assets["object"])
        copy_image(resolve(orig_row["image"]), case_assets["original"])
        copy_image(resolve(ft_row["image"]), case_assets["finetuned"])
        cases.append(
            {
                "id": item_id,
                "category": ft_row.get("category", manifest_row.get("category", "")),
                "original": {key: orig_row[key] for key in ["total", "object", "person", "artifact"]},
                "finetuned": {key: ft_row[key] for key in ["total", "object", "person", "artifact"]},
                "delta_total": round(ft_row["total"] - orig_row["total"], 6),
                "delta_object": round(ft_row["object"] - orig_row["object"], 6),
                "delta_person": round(ft_row["person"] - orig_row["person"], 6),
                "delta_artifact": round(ft_row["artifact"] - orig_row["artifact"], 6),
                "assets": {key: str(value) for key, value in case_assets.items()},
            }
        )

    metrics = {
        "original": summary_metrics(original_payload),
        "finetuned": summary_metrics(finetuned_payload),
        "best": summary_metrics(best_payload),
        "best_source_counts": best_payload.get("source_counts", {}),
        "fine_tuned_winner_count": len(cases),
    }
    payload = {
        "method": "Geo-Affordance Candidate Selection",
        "source_summaries": {
            "original": args.original_summary,
            "finetuned": args.finetuned_summary,
            "best": args.best_summary,
        },
        "metrics": metrics,
        "cases": cases,
    }
    write_json(output_dir / "winner_manifest.json", payload)
    render_html(output_dir, cases, metrics)
    render_method_doc(output_dir, cases, metrics)
    video_path, frame_paths = render_video(output_dir, cases, args.seconds_per_case)

    print(json.dumps(metrics, indent=2))
    print(f"Wrote winner manifest -> {output_dir / 'winner_manifest.json'}")
    print(f"Wrote UI -> {output_dir / 'index.html'}")
    print(f"Wrote method doc -> {output_dir / 'method.md'}")
    print(f"Wrote video -> {video_path}")
    print(f"Wrote {len(frame_paths)} frames -> {output_dir / 'frames'}")


if __name__ == "__main__":
    raise SystemExit(main())
