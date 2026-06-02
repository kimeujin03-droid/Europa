from __future__ import annotations

from pathlib import Path
import xml.etree.ElementTree as ET

import pandas as pd


PDS_NS = {"pds": "http://pds.nasa.gov/pds4/pds/v1"}


def _text(root: ET.Element, path: str, default: str = "") -> str:
    value = root.findtext(path, namespaces=PDS_NS)
    return value if value is not None else default


def parse_relab_xml(xml_path: str | Path) -> dict[str, str | int | float]:
    xml_path = Path(xml_path)
    root = ET.parse(xml_path).getroot()
    table = root.find(".//pds:Table_Character", PDS_NS)
    if table is None:
        raise ValueError(f"No Table_Character found in {xml_path}")

    file_name = _text(root, ".//pds:File/pds:file_name")
    records = int(_text(table, "pds:records"))
    offset = int(_text(table, "pds:offset"))
    record_length = int(_text(table, ".//pds:record_length"))

    fields = {}
    for field in table.findall(".//pds:Field_Character", PDS_NS):
        name = _text(field, "pds:name")
        fields[name] = {
            "location": int(_text(field, "pds:field_location")) - 1,
            "length": int(_text(field, "pds:field_length")),
            "unit": _text(field, "pds:unit"),
        }

    specimen_name = root.findtext(".//{http://pds.nasa.gov/pds4/speclib/v1}specimen_name") or ""
    specimen_description = root.findtext(".//{http://pds.nasa.gov/pds4/speclib/v1}specimen_description") or ""
    spectrum_id = _text(root, ".//pds:logical_identifier").split(":")[-1]

    return {
        "file_name": file_name,
        "records": records,
        "offset": offset,
        "record_length": record_length,
        "fields": fields,
        "specimen_name": specimen_name,
        "specimen_description": specimen_description,
        "spectrum_id": spectrum_id,
    }


def read_relab_tab(tab_path: str | Path, xml_path: str | Path) -> pd.DataFrame:
    tab_path = Path(tab_path)
    meta = parse_relab_xml(xml_path)
    fields = meta["fields"]

    wavelength_field = fields["Wavelength"]
    reflectance_field = fields["Reflectance"]
    rows = []
    with tab_path.open("r", encoding="utf-8", errors="replace") as handle:
        handle.seek(int(meta["offset"]))
        for _ in range(int(meta["records"])):
            line = handle.readline()
            if not line:
                break
            wavelength = line[
                wavelength_field["location"] : wavelength_field["location"] + wavelength_field["length"]
            ].strip()
            reflectance = line[
                reflectance_field["location"] : reflectance_field["location"] + reflectance_field["length"]
            ].strip()
            rows.append((float(wavelength), float(reflectance)))

    df = pd.DataFrame(rows, columns=["wavelength_um", "value"])
    if wavelength_field.get("unit", "").lower() == "nm":
        df["wavelength_um"] = df["wavelength_um"] / 1000.0
    return df
