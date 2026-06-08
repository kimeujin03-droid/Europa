#!/usr/bin/env python
"""Discover and screen Galileo/NIMS Europa mosaic cubes by hemisphere/terrain."""
from __future__ import annotations

import argparse
import importlib.util
import re
import sys
import urllib.request
from pathlib import Path
from urllib.parse import urljoin

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.config import ExperimentConfig
from src.endmember_loader import load_processed_endmembers
import src.forward_model as forward_model


PDS_BASE = "https://planetarydata.jpl.nasa.gov/img/data/go-e-nims-4-mosaic-v1.0/"
OUT_DIR = ROOT / "results" / "nims_candidate_screen"
RAW_DIR = ROOT / "data" / "raw" / "nims_pds"
GEOMAP_DIR = ROOT / "data" / "raw" / "usgs_europa_geomap"
STAGE_DIR = ROOT / "data" / "processed" / "endmembers_ablation" / "02_ocean_salt_usgs_epsomite"
SPEC_COLS = [f"spec_{j:03d}" for j in range(181)]


def load_nims20():
    path = ROOT / "scripts" / "20_nims_sanity_check.py"
    spec = importlib.util.spec_from_file_location("nims20", path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Cannot import {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def fetch_text(url: str, timeout: int = 60) -> str:
    req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
    with urllib.request.urlopen(req, timeout=timeout) as response:
        return response.read().decode("utf-8", errors="replace")


def fetch_bytes(url: str, timeout: int = 120) -> bytes:
    req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
    with urllib.request.urlopen(req, timeout=timeout) as response:
        return response.read()


def discover_cube_urls() -> pd.DataFrame:
    root_html = fetch_text(PDS_BASE)
    volumes = sorted(set(re.findall(r'href="(go_\d+/)"', root_html)))
    rows = []
    for volume in volumes:
        europa_url = urljoin(PDS_BASE, f"{volume}europa/")
        try:
            html = fetch_text(europa_url)
        except Exception as exc:
            rows.append({"volume": volume.strip("/"), "file": "", "url": europa_url, "discover_error": str(exc)})
            continue
        for href in sorted(set(re.findall(r'href="([^"]+\.qub)"', html, flags=re.IGNORECASE))):
            rows.append(
                {
                    "volume": volume.strip("/"),
                    "file": href.split("/")[-1],
                    "url": urljoin(europa_url, href),
                    "discover_error": "",
                }
            )
    return pd.DataFrame(rows)


def read_label_from_url(url: str) -> str:
    # Most QUB labels are in the first 60 KB. If the server ignores Range, this
    # still works; labels are extracted from the initial bytes.
    req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0", "Range": "bytes=0-65535"})
    with urllib.request.urlopen(req, timeout=120) as response:
        data = response.read(65536)
    text = data.decode("latin-1", errors="replace")
    end = re.search(r"\nEND\s*\n", text)
    return text[: end.end()] if end else text


def parse_float(label: str, key: str) -> float | None:
    match = re.search(rf"{re.escape(key)}\s*=\s*([-+]?\d+(?:\.\d+)?)", label)
    return float(match.group(1)) if match else None


def parse_int_tuple(label: str, key: str) -> tuple[int, ...] | None:
    match = re.search(rf"{re.escape(key)}\s*=\s*\(([^)]+)\)", label, flags=re.DOTALL)
    if not match:
        return None
    return tuple(int(x.strip()) for x in match.group(1).split(",") if x.strip())


def parse_band_centers(label: str) -> np.ndarray | None:
    match = re.search(r"BAND_BIN_CENTER\s*=\s*\(([^)]+)\)", label, flags=re.DOTALL)
    if not match:
        return None
    values = [float(x.strip()) for x in match.group(1).split(",") if x.strip()]
    return np.asarray(values, dtype=float)


def center_lon_w(label: str) -> float | None:
    start = parse_float(label, "START_SUB_SPACECRAFT_LONGITUDE")
    stop = parse_float(label, "STOP_SUB_SPACECRAFT_LONGITUDE")
    if start is not None and stop is not None:
        return (start + stop) / 2.0
    center = parse_float(label, "CENTER_LONGITUDE")
    if center is not None:
        return center
    return parse_float(label, "SUB_SPACECRAFT_LONGITUDE")


def hemisphere_from_lon(lon_w: float | None) -> str:
    if lon_w is None:
        return "unknown"
    lon = lon_w % 360.0
    return "trailing" if 180.0 <= lon < 360.0 else "leading"


def terrain_group(unit: str) -> str:
    text = (unit or "").strip().lower()
    if text in {"chm", "chh", "chl", "c", "ce"} or text.startswith("ch"):
        return "chaos"
    if text in {"pr", "cpr", "cpre"}:
        return "plains"
    return "other"


def load_geomap(nims20):
    try:
        return nims20.load_usgs_geomap(GEOMAP_DIR)
    except Exception as exc:
        print(f"[WARN] Could not load geomap: {exc}")
        return None


def lookup_unit(gdf, lon_w: float | None) -> str:
    if gdf is None or lon_w is None:
        return "__unknown__"
    try:
        features, unit = load_nims20().geo_lookup_center(lon_w, 1, gdf)
        return unit
    except Exception:
        return "__unknown__"


def metadata_table(max_labels: int | None = None) -> pd.DataFrame:
    nims20 = load_nims20()
    gdf = load_geomap(nims20)
    discovered = discover_cube_urls()
    discovered = discovered[discovered["file"].astype(str).str.len() > 0].reset_index(drop=True)
    if max_labels is not None:
        discovered = discovered.head(max_labels)

    rows = []
    for idx, row in discovered.iterrows():
        label_error = ""
        label = ""
        try:
            label = read_label_from_url(row["url"])
        except Exception as exc:
            label_error = str(exc)

        wave = parse_band_centers(label) if label else None
        lon_w = center_lon_w(label) if label else None
        unit = lookup_unit(gdf, lon_w)
        core_items = parse_int_tuple(label, "CORE_ITEMS") if label else None
        n_bands = int(core_items[2]) if core_items and len(core_items) >= 3 else (len(wave) if wave is not None else 0)
        wave_min = float(np.nanmin(wave)) if wave is not None and len(wave) else np.nan
        wave_max = float(np.nanmax(wave)) if wave is not None and len(wave) else np.nan
        coverage_0p7_2p0 = bool(wave is not None and wave_min <= 0.7 and wave_max >= 2.0)
        coverage_0p7_5p2 = bool(wave is not None and wave_min <= 0.7 and wave_max >= 5.2)
        hemi = hemisphere_from_lon(lon_w)
        terrain = terrain_group(unit)
        group = f"{hemi}_{terrain}" if hemi in {"leading", "trailing"} and terrain in {"chaos", "plains"} else "other"

        rows.append(
            {
                "volume": row["volume"],
                "file": row["file"],
                "url": row["url"],
                "center_lon_w": lon_w,
                "hemisphere": hemi,
                "geo_unit": unit,
                "terrain": terrain,
                "group": group,
                "n_bands": n_bands,
                "wave_min_um": wave_min,
                "wave_max_um": wave_max,
                "coverage_0p7_2p0": coverage_0p7_2p0,
                "coverage_0p7_5p2": coverage_0p7_5p2,
                "incidence_angle": parse_float(label, "INCIDENCE_ANGLE") if label else None,
                "emission_angle": parse_float(label, "EMISSION_ANGLE") if label else None,
                "phase_angle": parse_float(label, "PHASE_ANGLE") if label else None,
                "label_error": label_error,
            }
        )
        print(f"[{idx+1}/{len(discovered)}] {row['volume']}/{row['file']} -> {group} {wave_min:.2f}-{wave_max:.2f} um")
    return pd.DataFrame(rows)


def generate_synth(stage_dir: Path, n: int, mask: np.ndarray) -> np.ndarray:
    original_available = forward_model.processed_endmembers_available
    original_loader = forward_model.load_processed_endmembers
    try:
        forward_model.processed_endmembers_available = lambda: True
        forward_model.load_processed_endmembers = lambda wavelengths: load_processed_endmembers(
            wavelengths, base_dir=stage_dir
        )
        cfg = ExperimentConfig(seed=42, noise_condition="harsh", rad_simple_organic_hi=0.130)
        df = forward_model.generate_dataset(n=n, rho_geo=0.75, rho_rad=0.75, cfg=cfg)
    finally:
        forward_model.processed_endmembers_available = original_available
        forward_model.load_processed_endmembers = original_loader
    return df[SPEC_COLS].to_numpy()[:, mask]


def ood_ratio(nims: np.ndarray, synth: np.ndarray) -> tuple[float, float]:
    n_components = min(5, nims.shape[1], synth.shape[1], len(synth) - 1)
    pca = PCA(n_components=n_components, random_state=42)
    pca.fit(synth)
    synth_p = pca.transform(synth)
    nims_p = pca.transform(nims)
    dists = []
    for start in range(0, len(nims_p), 500):
        end = min(start + 500, len(nims_p))
        diff = nims_p[start:end, np.newaxis, :] - synth_p[np.newaxis, :, :]
        dists.extend(np.sqrt((diff**2).sum(axis=2)).min(axis=1))
    ref_left = synth_p[:500]
    ref_right = synth_p[500:1000]
    diff = ref_left[:, np.newaxis, :] - ref_right[np.newaxis, :, :]
    ref = float(np.sqrt((diff**2).sum(axis=2)).min(axis=1).mean())
    mean_dist = float(np.mean(dists))
    return mean_dist, mean_dist / (ref + 1e-9)


def download_cube(url: str, file_name: str) -> bytes:
    path = RAW_DIR / file_name
    if path.exists():
        return path.read_bytes()
    data = fetch_bytes(url, timeout=240)
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    path.write_bytes(data)
    return data


def screen_cubes(
    meta: pd.DataFrame,
    limit: int | None,
    max_per_group: int,
    n_synth: int,
    groups: list[str] | None = None,
) -> pd.DataFrame:
    nims20 = load_nims20()
    cfg = ExperimentConfig()
    mask = cfg.wavelengths <= 2.0 + 1e-9
    synth = generate_synth(STAGE_DIR, n_synth, mask)
    candidates = meta[
        meta["coverage_0p7_2p0"]
        & meta["group"].isin(["leading_chaos", "leading_plains", "trailing_chaos", "trailing_plains"])
    ].copy()
    if groups:
        candidates = candidates[candidates["group"].isin(groups)].copy()
    candidates["abs_lon_to_apex_or_anti"] = candidates["center_lon_w"].apply(
        lambda x: min(abs((x % 360) - 90), abs((x % 360) - 270)) if pd.notna(x) else 999
    )
    selected = candidates.sort_values(["group", "abs_lon_to_apex_or_anti"]).groupby("group", group_keys=False).head(max_per_group)
    if limit is not None:
        selected = selected.head(limit)

    rows = []
    for idx, row in selected.iterrows():
        try:
            data = download_cube(row["url"], row["file"])
            spectra_df, _ = nims20.read_nims_qub(data, cfg.wavelengths)
            n_valid = len(spectra_df)
            spec = spectra_df[SPEC_COLS].to_numpy()[:, mask]
            mean_ood, ratio = ood_ratio(spec, synth) if n_valid else (np.nan, np.nan)
            missing_frac = 1.0 - n_valid / max(1, int(np.prod(parse_int_tuple(nims20.extract_attached_label(data)[0], "CORE_ITEMS")[:2])))
            error = ""
        except Exception as exc:
            n_valid = 0
            mean_ood = np.nan
            ratio = np.nan
            missing_frac = np.nan
            error = str(exc)
        out = row.to_dict()
        out.update(
            {
                "n_valid_pixels": n_valid,
                "missing_frac": missing_frac,
                "ood_dist_0p7_2p0": mean_ood,
                "ood_ratio_0p7_2p0": ratio,
                "screen_pass": bool(n_valid >= 30 and pd.notna(ratio) and ratio <= 25.0),
                "screen_error": error,
            }
        )
        print(f"SCREEN {row['group']} {row['volume']}/{row['file']}: n={n_valid}, OOD={ratio:.1f}x {error}")
        rows.append(out)
    return pd.DataFrame(rows)


def plot_distribution(meta: pd.DataFrame, screened: pd.DataFrame | None = None) -> Path:
    counts = meta["group"].value_counts().rename_axis("group").reset_index(name="metadata_count")
    target_groups = ["leading_chaos", "leading_plains", "trailing_chaos", "trailing_plains"]
    counts = counts[counts["group"].isin(target_groups)].set_index("group").reindex(target_groups).fillna(0)
    if screened is not None and len(screened):
        pass_counts = screened[screened["screen_pass"]]["group"].value_counts()
        counts["screen_pass_count"] = pass_counts.reindex(target_groups).fillna(0).astype(int)
    else:
        counts["screen_pass_count"] = 0

    fig, ax = plt.subplots(figsize=(8, 4))
    x = np.arange(len(counts))
    width = 0.36
    ax.bar(x - width / 2, counts["metadata_count"], width=width, label="metadata eligible")
    ax.bar(x + width / 2, counts["screen_pass_count"], width=width, label="screen pass")
    ax.set_xticks(x)
    ax.set_xticklabels([g.replace("_", "\n") for g in counts.index])
    ax.set_ylabel("cube count")
    ax.set_title("NIMS Europa Candidate Cube Distribution")
    ax.legend()
    fig.tight_layout()
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    path = OUT_DIR / "nims_candidate_group_distribution.png"
    fig.savefig(path, dpi=220, bbox_inches="tight")
    plt.close(fig)
    return path


def main() -> None:
    args = parse_args()
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    meta_path = OUT_DIR / "nims_candidate_metadata.csv"
    if args.reuse_metadata and meta_path.exists():
        meta = pd.read_csv(meta_path)
    else:
        meta = metadata_table(max_labels=args.max_labels)
        meta.to_csv(meta_path, index=False)

    counts = (
        meta.groupby(["hemisphere", "terrain", "group"], dropna=False)
        .size()
        .reset_index(name="metadata_count")
        .sort_values(["hemisphere", "terrain"])
    )
    counts.to_csv(OUT_DIR / "nims_candidate_metadata_distribution.csv", index=False)
    print("\nMetadata distribution:")
    print(counts.to_string(index=False))

    screened = None
    if not args.metadata_only:
        groups = args.groups.split(",") if args.groups else None
        screened = screen_cubes(
            meta,
            limit=args.screen_limit,
            max_per_group=args.max_per_group,
            n_synth=args.n_synth,
            groups=groups,
        )
        screened.to_csv(OUT_DIR / "nims_candidate_screened.csv", index=False)
        screen_counts = (
            screened.groupby(["group", "screen_pass"], dropna=False)
            .size()
            .reset_index(name="count")
            .sort_values(["group", "screen_pass"])
        )
        screen_counts.to_csv(OUT_DIR / "nims_candidate_screened_distribution.csv", index=False)
        print("\nScreened distribution:")
        print(screen_counts.to_string(index=False))

    fig = plot_distribution(meta, screened)
    print(f"\nSaved {meta_path}")
    print(f"Saved {OUT_DIR / 'nims_candidate_metadata_distribution.csv'}")
    print(f"Saved {fig}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--metadata-only", action="store_true", help="Only discover labels and metadata distribution.")
    parser.add_argument("--reuse-metadata", action="store_true", help="Reuse existing metadata CSV if present.")
    parser.add_argument("--max-labels", type=int, help="Limit number of labels for a quick smoke test.")
    parser.add_argument("--screen-limit", type=int, default=24, help="Maximum cubes to download/screen.")
    parser.add_argument("--max-per-group", type=int, default=8, help="Maximum cubes to screen per group.")
    parser.add_argument("--n-synth", type=int, default=3000, help="Synthetic spectra for OOD screening.")
    parser.add_argument("--groups", help="Comma-separated groups to screen, e.g. leading_plains,trailing_chaos.")
    return parser.parse_args()


if __name__ == "__main__":
    main()
