#!/usr/bin/env python
from __future__ import annotations

from pathlib import Path
import re
import xml.etree.ElementTree as ET

import pandas as pd


INV = Path("data/raw/relab/collection_data_reflectance_inventory.csv")
RAW_RELAB = Path("data/raw/relab")

SEARCH_TERMS = {
    "ice": ["water ice", "h2o ice", "h2o-ice", "ice, water"],
    "ocean_salt": [
        "epsomite",
        "mirabilite",
        "bloedite",
        "magnesium sulfate",
        "sodium sulfate",
        "sulfate",
        "carbonate",
        "natron",
    ],
    "sulfuric_acid_hydrate": ["sulfuric acid", "sulphuric acid", "sulfuric acid hydrate", "h2so4"],
    "sulfur_so2": ["sulfur", "sulphur", "sulfur dioxide", "sulphur dioxide", "so2"],
    "h2o2": ["hydrogen peroxide", "h2o2", "peroxide"],
    "simple_organic": [
        "glycine",
        "alanine",
        "amino acid",
        "acetate",
        "carboxyl",
        "formate",
        "amide",
        "organic",
        "benzene",
        "toluene",
        "aliphatic",
        "aromatic",
    ],
}


def read_inventory() -> pd.DataFrame:
    rows = pd.read_csv(INV, header=None, names=["member_status", "lidvid"])
    rows["spectrum_id"] = rows["lidvid"].str.extract(r"data_reflectance:([^:]+)::", expand=False)
    rows["lid"] = rows["lidvid"].str.replace(r"::.*$", "", regex=True)
    return rows


def xml_text(xml_path: Path) -> str:
    try:
        root = ET.parse(xml_path).getroot()
    except ET.ParseError:
        return xml_path.read_text(encoding="utf-8", errors="replace").lower()
    chunks = [xml_path.name]
    for elem in root.iter():
        if elem.text and elem.text.strip():
            chunks.append(elem.text.strip())
    return " | ".join(chunks).lower()


def local_xml_index() -> pd.DataFrame:
    rows = []
    for xml_path in RAW_RELAB.rglob("*.xml"):
        if xml_path.name.startswith("collection_"):
            continue
        spectrum_id = xml_path.stem
        rows.append(
            {
                "spectrum_id": spectrum_id,
                "raw_xml": str(xml_path).replace("\\", "/"),
                "raw_tab": str(xml_path.with_suffix(".tab")).replace("\\", "/"),
                "metadata_text": xml_text(xml_path),
            }
        )
    return pd.DataFrame(rows)


def row_text(row: pd.Series) -> str:
    vals = []
    for value in row.values:
        if pd.notna(value):
            vals.append(str(value).lower())
    return " | ".join(vals)


def contains_term(texts: pd.Series, term: str) -> pd.Series:
    term = term.lower()
    if term == "organic":
        return texts.str.contains(r"(?<!in)organic", regex=True, na=False)
    if re.fullmatch(r"[a-z0-9]+", term):
        return texts.str.contains(rf"\b{re.escape(term)}\b", regex=True, na=False)
    return texts.str.contains(re.escape(term), regex=True, na=False)


def main() -> None:
    df = read_inventory()
    xml_df = local_xml_index()
    if not xml_df.empty:
        df = df.merge(xml_df, how="left", on="spectrum_id")
    else:
        df["metadata_text"] = ""
        df["raw_xml"] = ""
        df["raw_tab"] = ""

    print("Columns:")
    print(list(df.columns))
    print()
    print(f"Inventory rows: {len(df)}")
    print(f"Local XML metadata rows: {len(xml_df)}")
    if xml_df.empty:
        print("Note: inventory rows contain only LIDs. Download candidate XML metadata before expecting keyword hits.")

    texts = df.apply(row_text, axis=1)
    out_dir = Path("data/manifest/relab_candidates")
    out_dir.mkdir(parents=True, exist_ok=True)

    for group, terms in SEARCH_TERMS.items():
        mask = pd.Series(False, index=df.index)
        for term in terms:
            mask |= contains_term(texts, term)

        hits = df.loc[mask].copy()
        print(f"\n[{group}] hits = {len(hits)}")
        preview = hits.head(100)
        preview.to_csv(out_dir / f"{group}_candidates.csv", index=False, encoding="utf-8-sig")

        show_cols = [c for c in ["spectrum_id", "raw_xml", "raw_tab", "lidvid", "metadata_text"] if c in preview.columns]
        text = preview[show_cols].head(10).to_string(index=False)
        print(text.encode("utf-8", errors="replace").decode("utf-8", errors="replace"))


if __name__ == "__main__":
    main()
