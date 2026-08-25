"""
datasets/xdash/clinical_metadata.py
=====================================
Load and parse clinical metadata from the XDash Excel file.

Injury codes (dia_code column):
    0 = healthy
    1 = rotator_cuff
    2 = glenohumeral_arthritis
    3 = biceps_tendonitis
    4 = shoulder_bursitis
"""

from pathlib import Path

import pandas as pd


INJURY_NAMES = {
    0: "healthy",
    1: "rotator_cuff",
    2: "glenohumeral_arthritis",
    3: "biceps_tendonitis",
    4: "shoulder_bursitis",
}


class ClinicalMetadataLoader:
    """Load and parse clinical metadata from the XDash Excel file."""

    def __init__(self, excel_path: Path | str):
        self.excel_path = Path(excel_path)
        self.metadata: dict = {}

    def load(self) -> dict:
        """Load metadata from Excel; returns {patient_id: clinical_dict}."""
        if not self.excel_path.exists():
            print(f"Warning: clinical metadata file not found: {self.excel_path}")
            return {}

        df = self._read_excel()
        if df is None:
            return {}

        for _, row in df.iterrows():
            patient_id = row["id"]
            entry = self._parse_row(row)
            issues = _validate_entry(entry)
            if issues:
                print(f"Warning [{patient_id}]: {', '.join(issues)}")
            self.metadata[patient_id] = entry

        print(f"Loaded clinical metadata for {len(self.metadata)} subjects")
        return self.metadata

    def _read_excel(self) -> pd.DataFrame | None:
        for engine in ("openpyxl", "xlrd"):
            try:
                return pd.read_excel(self.excel_path, engine=engine)
            except Exception as e:
                print(f"Warning: Excel load failed (engine={engine}): {e}")
        return None

    def _parse_row(self, row) -> dict:
        dia_code = int(row["dia_code"])
        right = bool(row["dia_side_r"] == 1)
        left = bool(row["dia_side_l"] == 1)

        if right and left:
            laterality = "bilateral"
        elif right:
            laterality = "right"
        elif left:
            laterality = "left"
        else:
            laterality = "none"

        return {
            "age": int(row["age"]),
            "gender": "male" if row["sex"] == 1 else "female",
            "diagnosis_code": dia_code,
            "injury_type": INJURY_NAMES.get(dia_code, "unknown"),
            "laterality": laterality,
            "dominant_hand": "right" if row["hand_xr"] == 1 else "left",
            "affected_side_right": right,
            "affected_side_left": left,
            "baseline_scores": {
                "DASH": float(row["C_DASH"]),
                "QuickDASH": float(row["X_DASH"]),
                "DASH_combined": float(row["c and x"]),
            },
        }

    def get(self, patient_id: str) -> dict | None:
        return self.metadata.get(patient_id)

    @staticmethod
    def control_template() -> dict:
        """Return a default metadata dict for healthy controls."""
        return {
            "age": None,
            "gender": None,
            "diagnosis_code": 0,
            "injury_type": "healthy",
            "laterality": None,
            "dominant_hand": None,
            "affected_side_right": False,
            "affected_side_left": False,
            "baseline_scores": {},
        }


def load_clinical_metadata(excel_path: Path | str) -> dict:
    """Convenience wrapper — load metadata and return the dict."""
    return ClinicalMetadataLoader(excel_path).load()


# ---------------------------------------------------------------------------
# Internal validation (no external dependency)
# ---------------------------------------------------------------------------

def _validate_entry(entry: dict) -> list[str]:
    issues = []
    age = entry.get("age")
    if age is not None and not (0 <= age <= 120):
        issues.append(f"age {age} out of range")
    gender = entry.get("gender")
    if gender not in ("male", "female", None):
        issues.append(f"unexpected gender value {gender!r}")
    if not entry.get("injury_type"):
        issues.append("missing injury_type")
    return issues
