"""
datasets/xdash/io_utils.py
===========================
Robust I/O helpers for XDash raw CSV files.

Unity exports are sometimes UTF-16; load_csv_robust handles both encodings.
"""

from pathlib import Path

import pandas as pd


def load_csv_robust(
    file_path: Path | str,
    index_col: bool | int = False,
    header: str | int = "infer",
) -> pd.DataFrame | None:
    """Load a CSV with automatic UTF-8 / UTF-16 fallback.

    Returns None if the file is missing or both encodings fail.
    """
    path = Path(file_path)
    if not path.exists():
        print(f"Warning: file not found: {path}")
        return None

    for enc in ("utf-8", "utf-16"):
        try:
            return pd.read_csv(path, index_col=index_col, header=header, encoding=enc)
        except UnicodeDecodeError:
            continue
        except Exception as e:
            print(f"Error loading {path} ({enc}): {e}")
            return None

    print(f"Error: could not decode {path} as UTF-8 or UTF-16")
    return None


def safe_parse_hand(value) -> str | None:
    """Parse a hand field to 'left', 'right', or None."""
    if value is None:
        return None
    try:
        if pd.isna(value):
            return None
    except (TypeError, ValueError):
        pass

    s = str(value).lower()
    if "left" in s:
        return "left"
    if "right" in s:
        return "right"
    return None


def validate_dataframe(
    df: pd.DataFrame | None,
    required_columns: list[str] | None = None,
    min_rows: int = 1,
) -> tuple[bool, str | None]:
    """Check that a DataFrame meets basic requirements.

    Returns (is_valid, error_message).
    """
    if df is None:
        return False, "DataFrame is None"
    if len(df) < min_rows:
        return False, f"has {len(df)} rows; minimum is {min_rows}"
    if required_columns:
        missing = set(required_columns) - set(df.columns)
        if missing:
            return False, f"missing columns: {missing}"
    return True, None


def load_subject_files(subject_dir: Path | str) -> dict[str, pd.DataFrame | None]:
    """Load Master.csv, PlayerMovement.csv, and SurveyResponse.csv for a subject.

    Returns {'master': df|None, 'movement': df|None, 'survey': df|None}.
    """
    d = Path(subject_dir)
    return {
        "master": load_csv_robust(d / "Master.csv"),
        "movement": load_csv_robust(d / "PlayerMovement.csv"),
        "survey": load_csv_robust(d / "SurveyResponse.csv"),
    }
