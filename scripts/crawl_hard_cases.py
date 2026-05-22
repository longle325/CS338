#!/usr/bin/env python
import argparse
import json
import sys
import time
import urllib.parse
import urllib.request
import urllib.error
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from omnitry.enhance.data import write_json


COMMONS_API = "https://commons.wikimedia.org/w/api.php"

DEFAULT_QUERIES = {
    "ring": ["person wearing ring close up hand", "jewelry ring on finger"],
    "earrings": ["person wearing earrings close up", "earrings portrait hair occlusion"],
    "bracelet": ["bracelet on wrist person", "person wearing bracelet hand"],
    "watch": ["watch on wrist person", "wristwatch close up hand"],
    "glasses": ["person wearing glasses side view", "eyeglasses portrait"],
    "sunglasses": ["person wearing sunglasses side view", "sunglasses portrait"],
    "bag": ["person carrying shoulder bag", "person holding tote bag"],
    "shoe": ["person wearing shoes side view", "shoes on feet perspective"],
    "hat": ["person wearing hat profile", "hat on head portrait"],
}


def parse_args():
    parser = argparse.ArgumentParser(description="Crawl licensed hard-case image candidates from Wikimedia Commons.")
    parser.add_argument("--output", default="data/hard_cases/commons_hard_cases.json")
    parser.add_argument("--download-dir", default="data/hard_cases/commons_images")
    parser.add_argument("--max-per-query", type=int, default=5)
    parser.add_argument("--download", action="store_true")
    parser.add_argument("--sleep", type=float, default=1.0)
    return parser.parse_args()


def request_json(params):
    query = urllib.parse.urlencode(params)
    request = urllib.request.Request(
        f"{COMMONS_API}?{query}",
        headers={"User-Agent": "OmniTry-CS338-hard-case-crawler/0.1 (educational research)"},
    )
    with urllib.request.urlopen(request, timeout=30) as response:
        return json.loads(response.read().decode("utf-8"))


def search_commons(query, limit):
    params = {
        "action": "query",
        "format": "json",
        "generator": "search",
        "gsrnamespace": 6,
        "gsrsearch": query,
        "gsrlimit": limit,
        "prop": "imageinfo",
        "iiprop": "url|extmetadata|mime|size",
        "iiurlwidth": 768,
    }
    data = request_json(params)
    pages = data.get("query", {}).get("pages", {})
    return [pages[key] for key in sorted(pages)]


def clean_metadata(page, category, query):
    info = (page.get("imageinfo") or [{}])[0]
    ext = info.get("extmetadata") or {}

    def ext_value(name):
        value = ext.get(name, {})
        return value.get("value") if isinstance(value, dict) else None

    return {
        "id": f"commons_{page.get('pageid')}",
        "source": "wikimedia_commons",
        "category": category,
        "query": query,
        "title": page.get("title"),
        "url": info.get("url"),
        "thumb_url": info.get("thumburl"),
        "mime": info.get("mime"),
        "width": info.get("width"),
        "height": info.get("height"),
        "license": ext_value("LicenseShortName"),
        "license_url": ext_value("LicenseUrl"),
        "artist": ext_value("Artist"),
        "credit": ext_value("Credit"),
        "usage_terms": ext_value("UsageTerms"),
    }


def download_image(item, download_dir):
    url = item.get("thumb_url") or item.get("url")
    if not url:
        return None
    suffix = Path(urllib.parse.urlparse(url).path).suffix or ".jpg"
    filename = f"{item['id']}{suffix}"
    output_path = Path(download_dir) / item["category"] / filename
    output_path.parent.mkdir(parents=True, exist_ok=True)
    urllib.request.urlretrieve(url, output_path)
    item["local_path"] = str(output_path)
    return output_path


def main():
    args = parse_args()
    items = []
    errors = []
    seen_urls = set()

    for category, queries in DEFAULT_QUERIES.items():
        for query in queries:
            print(f"Searching Commons: [{category}] {query}")
            try:
                pages = search_commons(query, args.max_per_query)
            except urllib.error.HTTPError as exc:
                errors.append({"category": category, "query": query, "error": f"HTTP {exc.code}: {exc.reason}"})
                print(f"  skipped: HTTP {exc.code}: {exc.reason}")
                time.sleep(max(args.sleep, 2.0))
                continue
            except Exception as exc:
                errors.append({"category": category, "query": query, "error": str(exc)})
                print(f"  skipped: {exc}")
                time.sleep(args.sleep)
                continue

            for page in pages:
                item = clean_metadata(page, category, query)
                url = item.get("url")
                if not url or url in seen_urls:
                    continue
                seen_urls.add(url)
                if args.download:
                    try:
                        download_image(item, args.download_dir)
                    except Exception as exc:
                        item["download_error"] = str(exc)
                items.append(item)
            time.sleep(args.sleep)

    payload = {
        "source": "wikimedia_commons",
        "note": "Raw licensed image candidates for hard-case expansion. Human filtering is required before training.",
        "count": len(items),
        "errors": errors,
        "items": items,
    }
    write_json(Path(args.output), payload)
    print(f"Wrote {len(items)} crawled hard-case candidates -> {args.output}")


if __name__ == "__main__":
    raise SystemExit(main())
