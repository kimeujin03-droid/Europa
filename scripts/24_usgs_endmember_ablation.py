#!/usr/bin/env python
"""Build USGS hydrated-salt replacements and measure NIMS OOD by stage.

The script keeps the committed processed endmembers intact. For each ablation
stage it copies the baseline endmember set into
data/processed/endmembers_ablation/<stage>, replaces selected endmembers, then
generates synthetic spectra from that temporary library and measures the NIMS
OOD ratio at 0.7-5.2 um and 0.7-2.0 um.

USGS SPLIB07 contains hydrated sulfate salts such as epsomite and mirabilite,
but not a tabulated hydrated sulfuric-acid spectrum in this local archive. Use
--acid-csv when Carlson/Loeffler/digitized HSO data are available.
"""
from __future__ import annotations

import argparse
import re
import shutil
import sys
import zipfile
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.decomposition import PCA

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import src.forward_model as forward_model
from src.config import ExperimentConfig
from src.endmember_loader import ENDMEMBER_FILES, load_processed_endmembers
from src.spectral_preprocess import interpolate_to_grid


USGS_ROOT = ROOT / "data" / "raw" / "usgs_splib07"
USGS_DIR = USGS_ROOT / "ASCIIdata_splib07a"
USGS_ZIP = USGS_ROOT / "ASCIIdata_splib07a.zip"
CANDIDATE_DIR = USGS_ROOT / "processed_candidates"
BASE_ENDMEMBER_DIR = ROOT / "data" / "processed" / "endmembers"
ABLATION_DIR = ROOT / "data" / "processed" / "endmembers_ablation"
QC_DIR = ROOT / "results" / "qc"
NIMS_DIR = ROOT / "results" / "nims_sanity_check"

SPEC_COLS = [f"spec_{j:03d}" for j in range(181)]
NIMS_FILES = {
    "E6": NIMS_DIR / "nims_e6_scores.csv",
    "E15": NIMS_DIR / "nims_e15_scores.csv",
    "G2": NIMS_DIR / "nims_g2_scores.csv",
    "E11": NIMS_DIR / "nims_e11_scores.csv",
}

WAVELENGTH_FILES = {
    "ASD": "splib07a_Wavelengths_ASD_0.35-2.5_microns_2151_ch.txt",
    "BECK": "splib07a_Wavelengths_BECK_Beckman_0.2-3.0_microns.txt",
    "NIC4": "splib07a_Wavelengths_NIC4_Nicolet_1.12-216microns.txt",
}


@dataclass(frozen=True)
class UsgsPart:
    source_rel: str
    wavelength_key: str


@dataclass(frozen=True)
class UsgsCandidate:
    candidate_id: str
    source_name: str
    parts: tuple[UsgsPart, ...]
    target_endmember: str = "ocean_salt"


HYDRATED_SALT_CANDIDATES = {
    "usgs_ammonium_chloride": UsgsCandidate(
        candidate_id="usgs_ammonium_chloride",
        source_name="USGS Ammonium Chloride GDS77",
        parts=(
            UsgsPart("ChapterA_ArtificialMaterials/splib07a_Ammonium_Chloride_GDS77_BECKa_AREF.txt", "BECK"),
        ),
    ),
    "usgs_epsomite": UsgsCandidate(
        candidate_id="usgs_epsomite",
        source_name="USGS Epsomite GDS149 hydrated magnesium sulfate",
        parts=(
            UsgsPart("ChapterM_Minerals/splib07a_Epsomite_GDS149_BECKc_AREF.txt", "BECK"),
            UsgsPart("ChapterM_Minerals/splib07a_Epsomite_GDS149_NIC4ccu_RREF.txt", "NIC4"),
        ),
    ),
    "usgs_mirabilite": UsgsCandidate(
        candidate_id="usgs_mirabilite",
        source_name="USGS Mirabilite GDS150 Na2SO4 hydrate",
        parts=(
            UsgsPart("ChapterM_Minerals/splib07a_Mirabilite_GDS150_Na2SO4_BECKa_AREF.txt", "BECK"),
            UsgsPart("ChapterM_Minerals/splib07a_Mirabilite_GDS150_Na2SO4_NIC4aa_RREF.txt", "NIC4"),
        ),
    ),
    "usgs_gypsum": UsgsCandidate(
        candidate_id="usgs_gypsum",
        source_name="USGS Gypsum HS333.3B sulfate hydrate",
        parts=(
            UsgsPart("ChapterM_Minerals/splib07a_Gypsum_HS333.3B_(Selenite)_BECKa_AREF.txt", "BECK"),
            UsgsPart("ChapterM_Minerals/splib07a_Gypsum_HS333.3B_(Selenite)_NIC4aaa_RREF.txt", "NIC4"),
        ),
    ),
    "usgs_halite": UsgsCandidate(
        candidate_id="usgs_halite",
        source_name="USGS Halite HS433.3B sodium chloride",
        parts=(
            UsgsPart("ChapterM_Minerals/splib07a_Halite_HS433.3B_BECKa_AREF.txt", "BECK"),
            UsgsPart("ChapterM_Minerals/splib07a_Halite_HS433.3B_NIC4aau_RREF.txt", "NIC4"),
        ),
    ),
    "usgs_polyhalite": UsgsCandidate(
        candidate_id="usgs_polyhalite",
        source_name="USGS Polyhalite NMNH92669-4 hydrated sulfate salt",
        parts=(
            UsgsPart("ChapterM_Minerals/splib07a_Polyhalite_NMNH92669-4_BECKa_AREF.txt", "BECK"),
            UsgsPart("ChapterM_Minerals/splib07a_Polyhalite_NMNH92669-4_NIC4aau_RREF.txt", "NIC4"),
        ),
    ),
    "usgs_potassium_sulfate": UsgsCandidate(
        candidate_id="usgs_potassium_sulfate",
        source_name="USGS Potassium Sulfate JT1-3282",
        parts=(
            UsgsPart("ChapterA_ArtificialMaterials/splib07a_Potassium_Sulfate_JT1-3282_ASDHRa_AREF.txt", "ASD"),
        ),
    ),
    "usgs_sodium_sulfate": UsgsCandidate(
        candidate_id="usgs_sodium_sulfate",
        source_name="USGS Sodium Sulfate JT1-3989",
        parts=(
            UsgsPart("ChapterA_ArtificialMaterials/splib07a_Sodium_Sulfate_JT1-3989_ASDHRc_AREF.txt", "ASD"),
        ),
    ),
}


def read_usgs_ascii(path: Path) -> pd.Series:
    values = []
    with path.open("r", encoding="utf-8", errors="replace") as handle:
        next(handle)
        for line in handle:
            text = line.strip()
            if not text:
                continue
            value = float(text)
            values.append(np.nan if value < -1e20 else value)
    return pd.Series(values, dtype=float)


def ensure_usgs_file(rel_path: str) -> Path:
    path = USGS_DIR / rel_path
    if path.exists():
        return path
    if not USGS_ZIP.exists():
        raise FileNotFoundError(f"Missing {path} and {USGS_ZIP}")
    zip_member = f"ASCIIdata_splib07a/{rel_path}"
    with zipfile.ZipFile(USGS_ZIP) as archive:
        try:
            archive.getinfo(zip_member)
        except KeyError as exc:
            raise FileNotFoundError(f"{zip_member} not found in {USGS_ZIP}") from exc
        path.parent.mkdir(parents=True, exist_ok=True)
        with archive.open(zip_member) as source, path.open("wb") as target:
            shutil.copyfileobj(source, target)
    return path


def extract_candidate(candidate: UsgsCandidate) -> Path:
    frames = []
    for part in candidate.parts:
        spectrum_path = ensure_usgs_file(part.source_rel)
        wave_path = ensure_usgs_file(WAVELENGTH_FILES[part.wavelength_key])
        spectrum = read_usgs_ascii(spectrum_path)
        wavelengths = read_usgs_ascii(wave_path)
        n = min(len(spectrum), len(wavelengths))
        frame = pd.DataFrame(
            {
                "wavelength_um": wavelengths.iloc[:n].to_numpy(),
                "value": spectrum.iloc[:n].to_numpy(),
                "instrument": part.wavelength_key,
            }
        )
        frames.append(frame)

    merged = pd.concat(frames, ignore_index=True)
    merged = merged.dropna(subset=["wavelength_um", "value"])
    merged = merged[(merged["wavelength_um"] > 0) & (merged["value"] > -1e20)]
    merged = merged.sort_values(["wavelength_um", "instrument"])
    # Prefer NIC4 in overlap because it carries the 3-5.2 um range used in OOD.
    merged["instrument_rank"] = merged["instrument"].map({"BECK": 0, "ASD": 1, "NIC4": 2}).fillna(0)
    merged = merged.sort_values(["wavelength_um", "instrument_rank"])
    merged = merged.drop_duplicates("wavelength_um", keep="last")
    out = merged[["wavelength_um", "value"]].copy()
    out["kind"] = "reflectance"
    out["source_id"] = candidate.candidate_id
    out["source_name"] = candidate.source_name
    out["notes"] = "USGS Spectral Library v7 composite extraction for endmember ablation"
    CANDIDATE_DIR.mkdir(parents=True, exist_ok=True)
    out_path = CANDIDATE_DIR / f"{candidate.candidate_id}.csv"
    out.to_csv(out_path, index=False)
    return out_path


def candidate_from_csv(path: Path, target_endmember: str, candidate_id: str | None = None) -> UsgsCandidate:
    return UsgsCandidate(
        candidate_id=candidate_id or path.stem,
        source_name=f"External replacement CSV: {path.name}",
        parts=(),
        target_endmember=target_endmember,
    )


def copy_baseline(stage_dir: Path) -> None:
    stage_dir.mkdir(parents=True, exist_ok=True)
    for filename in ENDMEMBER_FILES.values():
        source = BASE_ENDMEMBER_DIR / filename
        if not source.exists():
            raise FileNotFoundError(f"Missing baseline endmember: {source}")
        shutil.copy2(source, stage_dir / filename)


def write_replacement(stage_dir: Path, target_endmember: str, source_csv: Path, source_label: str) -> None:
    cfg = ExperimentConfig()
    wavelengths = cfg.wavelengths
    baseline_path = stage_dir / ENDMEMBER_FILES[target_endmember]
    baseline = interpolate_to_grid(pd.read_csv(baseline_path), wavelengths)
    source_df = pd.read_csv(source_csv)
    source_values = interpolate_to_grid(source_df, wavelengths)

    min_wave = float(source_df["wavelength_um"].min())
    max_wave = float(source_df["wavelength_um"].max())
    real_mask = (wavelengths >= min_wave) & (wavelengths <= max_wave)
    values = baseline.copy()
    values[real_mask] = source_values[real_mask]

    out = pd.DataFrame(
        {
            "wavelength_um": wavelengths,
            "value": values,
            "kind": "reflectance",
            "source_id": source_csv.stem,
            "source_name": source_label,
            "notes": f"Replacement for {target_endmember}; baseline retained outside measured coverage",
        }
    )
    out.to_csv(baseline_path, index=False)


def refresh_rad_salt(stage_dir: Path) -> None:
    cfg = ExperimentConfig()
    wavelengths = cfg.wavelengths
    salt = interpolate_to_grid(pd.read_csv(stage_dir / "ocean_salt.csv"), wavelengths)
    acid = interpolate_to_grid(pd.read_csv(stage_dir / "sulfuric_acid_hydrate.csv"), wavelengths)
    values = 0.5 * salt + 0.5 * acid
    out = pd.DataFrame(
        {
            "wavelength_um": wavelengths,
            "value": values,
            "kind": "reflectance_proxy",
            "source_id": "derived_from_stage_ocean_salt_and_sulfuric_acid_hydrate",
            "source_name": "stage radiolytic salt proxy",
            "notes": "Recomputed as 0.5 ocean_salt + 0.5 sulfuric_acid_hydrate for this ablation stage",
        }
    )
    out.to_csv(stage_dir / "rad_salt_proxy.csv", index=False)


def load_nims(mask: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    selected_cols = [col for col, include in zip(SPEC_COLS, mask) if include]
    parts = []
    labels = []
    for index, (_, path) in enumerate(NIMS_FILES.items()):
        if not path.exists():
            raise FileNotFoundError(f"Missing NIMS score file: {path}")
        arr = pd.read_csv(path, usecols=selected_cols).to_numpy()
        parts.append(arr)
        labels.extend([index] * len(arr))
    return np.vstack(parts), np.asarray(labels)


def ood_distances(nims: np.ndarray, synth: np.ndarray, n_components: int = 5) -> np.ndarray:
    n_components = min(n_components, synth.shape[1], len(synth) - 1)
    pca = PCA(n_components=n_components, random_state=42)
    pca.fit(synth)
    synth_proj = pca.transform(synth)
    nims_proj = pca.transform(nims)
    distances = np.empty(len(nims_proj), dtype=float)
    chunk = 500
    for start in range(0, len(nims_proj), chunk):
        end = min(start + chunk, len(nims_proj))
        diff = nims_proj[start:end, np.newaxis, :] - synth_proj[np.newaxis, :, :]
        distances[start:end] = np.sqrt((diff**2).sum(axis=2)).min(axis=1)
    return distances


def generate_synth_from_stage(stage_dir: Path, n: int, seed: int, mask: np.ndarray) -> np.ndarray:
    original_available = forward_model.processed_endmembers_available
    original_loader = forward_model.load_processed_endmembers
    try:
        forward_model.processed_endmembers_available = lambda: True
        forward_model.load_processed_endmembers = lambda wavelengths: load_processed_endmembers(
            wavelengths, base_dir=stage_dir
        )
        cfg = ExperimentConfig(seed=seed, noise_condition="harsh", rad_simple_organic_hi=0.130)
        df = forward_model.generate_dataset(n=n, rho_geo=0.75, rho_rad=0.75, cfg=cfg)
    finally:
        forward_model.processed_endmembers_available = original_available
        forward_model.load_processed_endmembers = original_loader
    return df[SPEC_COLS].to_numpy()[:, mask]


def measure_stage(
    stage_name: str,
    stage_dir: Path,
    n_synth: int,
    seed: int,
    band_ranges: list[tuple[float, float]] | None = None,
) -> list[dict]:
    cfg = ExperimentConfig()
    band_ranges = band_ranges or [(0.7, 5.2), (0.7, 2.0)]
    rows = []
    for wave_min, wave_max in band_ranges:
        mask = (cfg.wavelengths >= wave_min - 1e-9) & (cfg.wavelengths <= wave_max + 1e-9)
        nims, labels = load_nims(mask)
        synth = generate_synth_from_stage(stage_dir, n=n_synth, seed=seed, mask=mask)
        dists = ood_distances(nims, synth)
        synth_ref = ood_distances(synth[:500], synth[500:1000])
        row = {
            "stage": stage_name,
            "wave_min_um": wave_min,
            "wave_max_um": wave_max,
            "n_bands": int(mask.sum()),
            "synth_internal_ood": float(synth_ref.mean()),
            "mean_ood_dist": float(dists.mean()),
            "mean_ood_ratio": float(dists.mean() / (synth_ref.mean() + 1e-9)),
        }
        for index, enc in enumerate(NIMS_FILES):
            sub = dists[labels == index]
            row[f"ood_dist_{enc}"] = float(sub.mean())
            row[f"ood_ratio_{enc}"] = float(sub.mean() / (synth_ref.mean() + 1e-9))
        rows.append(row)
    return rows


def safe_stage_name(name: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", name).strip("_")


def stage_from_replacements(
    stage_name: str,
    replacements: list[tuple[str, Path, str]],
    n_synth: int,
    seed: int,
    band_ranges: list[tuple[float, float]] | None = None,
) -> list[dict]:
    stage_dir = ABLATION_DIR / stage_name
    copy_baseline(stage_dir)
    for target_name, source_path, label in replacements:
        write_replacement(stage_dir, target_name, source_path, label)
    refresh_rad_salt(stage_dir)
    return measure_stage(stage_name, stage_dir, n_synth, seed, band_ranges=band_ranges)


def build_and_measure_stages(args: argparse.Namespace) -> pd.DataFrame:
    replacements: list[tuple[str, str, Path, str]] = []
    band_ranges = parse_band_ranges(args.band_ranges)

    salt_ids = args.salt_candidates or ["usgs_epsomite"]
    for candidate_id in salt_ids:
        if candidate_id not in HYDRATED_SALT_CANDIDATES:
            raise ValueError(
                f"Unknown salt candidate {candidate_id!r}. "
                f"Choices: {', '.join(sorted(HYDRATED_SALT_CANDIDATES))}"
            )
        candidate = HYDRATED_SALT_CANDIDATES[candidate_id]
        csv_path = extract_candidate(candidate)
        replacements.append((candidate_id, "ocean_salt", csv_path, candidate.source_name))

    if args.acid_csv:
        acid_csv = Path(args.acid_csv)
        if not acid_csv.is_absolute():
            acid_csv = ROOT / acid_csv
        replacements.insert(
            0,
            (
                safe_stage_name(args.acid_id or acid_csv.stem),
                "sulfuric_acid_hydrate",
                acid_csv,
                f"External sulfuric-acid-hydrate replacement: {acid_csv.name}",
            ),
        )

    if args.tholin_csv:
        tholin_csv = Path(args.tholin_csv)
        if not tholin_csv.is_absolute():
            tholin_csv = ROOT / tholin_csv
        replacements.append(
            (
                safe_stage_name(args.tholin_id or tholin_csv.stem),
                "tholin_pah",
                tholin_csv,
                f"External tholin/PAH replacement: {tholin_csv.name}",
            )
        )

    rows = []
    baseline_dir = ABLATION_DIR / "00_baseline"
    copy_baseline(baseline_dir)
    refresh_rad_salt(baseline_dir)
    rows.extend(measure_stage("00_baseline", baseline_dir, args.n_synth, args.seed, band_ranges=band_ranges))

    if args.salt_mode == "individual":
        for index, (candidate_id, target, csv_path, source_label) in enumerate(replacements, start=1):
            stage_name = safe_stage_name(f"{index:02d}_{target}_{candidate_id}")
            rows.extend(
                stage_from_replacements(
                    stage_name,
                    [(target, csv_path, source_label)],
                    args.n_synth,
                    args.seed,
                    band_ranges=band_ranges,
                )
            )
    else:
        cumulative: list[tuple[str, Path, str, str]] = []
        for index, (candidate_id, target, csv_path, source_label) in enumerate(replacements, start=1):
            cumulative.append((target, csv_path, source_label, candidate_id))
            stage_name = safe_stage_name(
                f"{index:02d}_" + "_".join(f"{target}_{cid}" for target, _, _, cid in cumulative)
            )
            rows.extend(
                stage_from_replacements(
                    stage_name,
                    [(target_name, source_path, label) for target_name, source_path, label, _ in cumulative],
                    args.n_synth,
                    args.seed,
                    band_ranges=band_ranges,
                )
            )

    return pd.DataFrame(rows)


def list_candidates() -> None:
    print("Built-in hydrated salt candidates:")
    for candidate_id, candidate in HYDRATED_SALT_CANDIDATES.items():
        print(f"  {candidate_id}: {candidate.source_name}")
    print()
    print("Note: this local USGS SPLIB07 archive has hydrated sulfate salts,")
    print("but no hydrated sulfuric-acid endmember. Pass --acid-csv for HSO data.")


def parse_band_ranges(text: str | None) -> list[tuple[float, float]] | None:
    if not text:
        return None
    ranges = []
    for item in text.split(","):
        item = item.strip()
        if not item:
            continue
        try:
            lo_text, hi_text = item.split("-", maxsplit=1)
            lo = float(lo_text)
            hi = float(hi_text)
        except ValueError as exc:
            raise ValueError(f"Invalid band range {item!r}; expected 'lo-hi,lo-hi'.") from exc
        if hi <= lo:
            raise ValueError(f"Invalid band range {item!r}; upper bound must exceed lower bound.")
        ranges.append((lo, hi))
    if not ranges:
        raise ValueError("No valid band ranges supplied.")
    return ranges


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--list-candidates", action="store_true", help="Print built-in USGS candidates and exit.")
    parser.add_argument(
        "--salt-candidates",
        nargs="*",
        choices=sorted(HYDRATED_SALT_CANDIDATES),
        help="Hydrated salt candidate IDs to test. Default: usgs_epsomite.",
    )
    parser.add_argument(
        "--salt-mode",
        choices=["individual", "cumulative"],
        default="individual",
        help="Use each salt candidate as its own stage, or apply candidates cumulatively.",
    )
    parser.add_argument("--acid-csv", help="Optional sulfuric_acid_hydrate replacement CSV.")
    parser.add_argument("--acid-id", help="Short ID for the acid replacement stage.")
    parser.add_argument("--tholin-csv", help="Optional tholin_pah replacement CSV.")
    parser.add_argument("--tholin-id", help="Short ID for the tholin replacement stage.")
    parser.add_argument("--n-synth", type=int, default=4000, help="Synthetic spectra per stage.")
    parser.add_argument("--seed", type=int, default=42, help="Synthetic generation seed.")
    parser.add_argument(
        "--band-ranges",
        help="Comma-separated wavelength ranges, e.g. '0.7-2.0,2.0-2.5,2.5-3.5,3.5-4.1,4.1-5.2'.",
    )
    parser.add_argument(
        "--out",
        default=str(QC_DIR / "endmember_replacement_ood_ablation.csv"),
        help="Output CSV path.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.list_candidates:
        list_candidates()
        return
    QC_DIR.mkdir(parents=True, exist_ok=True)
    ABLATION_DIR.mkdir(parents=True, exist_ok=True)
    result = build_and_measure_stages(args)
    out_path = Path(args.out)
    if not out_path.is_absolute():
        out_path = ROOT / out_path
    out_path.parent.mkdir(parents=True, exist_ok=True)
    result.to_csv(out_path, index=False)
    print(result[["stage", "wave_max_um", "mean_ood_ratio"]].to_string(index=False))
    print(f"\nSaved {out_path}")


if __name__ == "__main__":
    main()
