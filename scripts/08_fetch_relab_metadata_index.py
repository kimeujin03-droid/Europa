#!/usr/bin/env python
from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from urllib.error import HTTPError, URLError
from urllib.request import urlopen
from urllib.parse import urljoin

from bs4 import BeautifulSoup


BASE = "https://pds-geosciences.wustl.edu/speclib/urn-nasa-pds-relab/data_reflectance"
SUBDIRS = ["bdr2", "bdr3", "ftir1", "ftir2"]
OUT = Path("data/raw/relab/metadata_cache")


def list_xml_urls(subdir: str) -> list[tuple[str, str]]:
    url = f"{BASE}/{subdir}/"
    with urlopen(url, timeout=60) as response:
        html = response.read()
    soup = BeautifulSoup(html, "html.parser")
    urls = []
    for link in soup.find_all("a"):
        href = link.get("href", "")
        name = Path(href).name
        if name.endswith(".xml") and not name.startswith("collection_"):
            urls.append((subdir, urljoin(url, href)))
    return urls


def fetch_one(item: tuple[str, str]) -> tuple[str, bool, str]:
    subdir, url = item
    target = OUT / subdir / Path(url).name
    if target.exists() and target.stat().st_size > 0:
        return str(target), True, "exists"
    target.parent.mkdir(parents=True, exist_ok=True)
    try:
        with urlopen(url, timeout=60) as response:
            target.write_bytes(response.read())
        return str(target), True, "downloaded"
    except HTTPError as exc:
        return str(target), False, f"http_{exc.code}"
    except URLError as exc:
        return str(target), False, f"url_error:{exc.reason}"
    except Exception as exc:
        return str(target), False, str(exc)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--subdir", choices=SUBDIRS, action="append", default=None)
    parser.add_argument("--limit", type=int, default=0, help="Download only the first N metadata files; 0 means all.")
    parser.add_argument("--workers", type=int, default=4)
    args = parser.parse_args()

    OUT.mkdir(parents=True, exist_ok=True)
    all_urls = []
    subdirs = args.subdir or SUBDIRS
    for subdir in subdirs:
        urls = list_xml_urls(subdir)
        print(f"{subdir}: {len(urls)} XML metadata files")
        all_urls.extend(urls)
    if args.limit > 0:
        all_urls = all_urls[: args.limit]
        print(f"Limit enabled: downloading first {len(all_urls)} metadata files")

    ok = 0
    failed = 0
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        futures = [pool.submit(fetch_one, item) for item in all_urls]
        for i, future in enumerate(as_completed(futures), start=1):
            path, success, status = future.result()
            ok += int(success)
            failed += int(not success)
            if i % 500 == 0 or not success:
                print(f"{i}/{len(futures)} ok={ok} failed={failed} last={Path(path).name} {status}")

    print(f"Done. ok={ok} failed={failed} out={OUT}")


if __name__ == "__main__":
    main()
