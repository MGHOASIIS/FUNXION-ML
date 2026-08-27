"""
Data paradigm selection.

ParadigmSelector is driven entirely by the dataset config loaded from
datasets/{name}/dataset.yaml.  No hardcoded task/paradigm logic lives here.

Works with both data formats:
  - Subject-level:  key = subject_id         (e.g. "PX01", "fx01")
  - Event-window:   key = window_id          (e.g. "fx07_task4_trial1")
Subject identity is always extracted via extract_subject_id(key).
"""
from typing import Dict, Tuple
import pandas as pd


def extract_subject_id(key: str) -> str:
    """Extract the subject identifier from a dict key.

    "PX01"              -> "PX01"
    "fx07_task4_trial1" -> "fx07"
    """
    return key.split("_")[0]


class ParadigmSelector:
    """Config-driven paradigm selection. Pass the loaded dataset_config dict."""

    def __init__(self, dataset_config: dict):
        self.config = dataset_config
        self.paradigm_configs = {
            int(k): v for k, v in dataset_config.get("paradigms", {}).items()
        }
        excl = dataset_config.get("exclude_subjects", {})
        self.exclude_g1 = set(excl.get("g1", []))
        self.exclude_g0 = set(excl.get("g0", []))
        self._metadata_df = None

        if self._any_needs_metadata():
            self._load_metadata()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def select_paradigm(
        self,
        patient_data: Dict,
        control_data: Dict,
        paradigm: int,
    ) -> Tuple[Dict, Dict]:
        patient_data = {
            k: v for k, v in patient_data.items()
            if extract_subject_id(k) not in self.exclude_g1
        }
        control_data = {
            k: v for k, v in control_data.items()
            if extract_subject_id(k) not in self.exclude_g0
        }
        if self.exclude_g1 or self.exclude_g0:
            print(f"[ParadigmSelector] Excluded g1={list(self.exclude_g1)}  "
                  f"g0={list(self.exclude_g0)}")

        pcfg = self.paradigm_configs.get(paradigm)
        if pcfg is None:
            available = sorted(self.paradigm_configs.keys())
            raise ValueError(
                f"Paradigm {paradigm} not in dataset config. Available: {available}"
            )

        g1 = self._apply_filter(patient_data, control_data, pcfg["g1_filter"])
        g0 = self._apply_filter(patient_data, control_data, pcfg["g0_filter"])

        print(f"Paradigm {paradigm}: {pcfg.get('name', '')}")
        print(f"  Group 1: {len(g1)}")
        print(f"  Group 0: {len(g0)}")
        return g1, g0

    def select_labels(
        self,
        patient_data: Dict,
        control_data: Dict,
        paradigm: int,
    ) -> Dict[str, Dict]:
        """Multi-label counterpart of select_paradigm().

        Returns one subject-data dict per named label in the paradigm's
        `labels` map (see dataset.yaml's `type: multilabel` schema). Unlike
        g1/g0, a subject may appear in more than one returned group — that's
        the point of multi-label classification.
        """
        patient_data = {
            k: v for k, v in patient_data.items()
            if extract_subject_id(k) not in self.exclude_g1
        }
        control_data = {
            k: v for k, v in control_data.items()
            if extract_subject_id(k) not in self.exclude_g0
        }

        pcfg = self.paradigm_configs.get(paradigm)
        if pcfg is None:
            available = sorted(self.paradigm_configs.keys())
            raise ValueError(
                f"Paradigm {paradigm} not in dataset config. Available: {available}"
            )
        if pcfg.get("type") != "multilabel":
            raise ValueError(
                f"Paradigm {paradigm} is not type: multilabel (got {pcfg.get('type', 'binary')!r})"
            )

        groups = {
            label_name: self._apply_filter(patient_data, control_data, flt)
            for label_name, flt in pcfg["labels"].items()
        }

        print(f"Paradigm {paradigm}: {pcfg.get('name', '')} (multilabel)")
        for label_name, group in groups.items():
            print(f"  {label_name}: {len(group)}")
        return groups

    def get_paradigm_type(self, paradigm: int) -> str:
        """Return 'binary' (default) or 'multilabel' for a paradigm id."""
        pcfg = self.paradigm_configs.get(paradigm)
        if pcfg is None:
            available = sorted(self.paradigm_configs.keys())
            raise ValueError(
                f"Paradigm {paradigm} not in dataset config. Available: {available}"
            )
        return pcfg.get("type", "binary")

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _any_needs_metadata(self) -> bool:
        for pcfg in self.paradigm_configs.values():
            sides = list(pcfg.get("labels", {}).values())
            for side_key in ("g1_filter", "g0_filter"):
                if side_key in pcfg:
                    sides.append(pcfg[side_key])
            for f in sides:
                if isinstance(f, dict) and f.get("filter") in ("metadata", "metadata_exclude"):
                    return True
        return False

    def _load_metadata(self):
        from config.paths import get_metadata_path
        meta_file = self.config.get("metadata_file")
        if not meta_file:
            return
        path = get_metadata_path(self.config["name"], meta_file)
        if not path.exists():
            raise FileNotFoundError(
                f"Metadata file not found: {path}\n"
                f"Place {meta_file} in storage/raw/{self.config['name']}/"
            )
        # Auto-detect header row — xlsx may have notes rows above the real header.
        needed_cols = self._needed_metadata_columns()
        for header_row in range(6):
            df = pd.read_excel(path, sheet_name="Sheet1", header=header_row)
            if any(c in df.columns for c in needed_cols):
                if header_row != 0:
                    print(f"[ParadigmSelector] Metadata header found at row {header_row}")
                self._metadata_df = df
                return
        raise ValueError(
            f"Could not find columns {needed_cols} in first 6 rows of {path}"
        )

    def _needed_metadata_columns(self) -> list:
        cols = []
        for pcfg in self.paradigm_configs.values():
            sides = list(pcfg.get("labels", {}).values())
            for side_key in ("g1_filter", "g0_filter"):
                if side_key in pcfg:
                    sides.append(pcfg[side_key])
            for f in sides:
                if isinstance(f, dict) and "column" in f:
                    cols.append(f["column"])
        return cols

    def _get_metadata_ids(self, column: str, values: list) -> set:
        if self._metadata_df is None:
            raise ValueError("Metadata filter requires metadata_file in dataset config")
        mask = self._metadata_df[column].isin(values)
        return set(self._metadata_df.loc[mask, "id"].tolist())

    def _apply_filter(self, patient_data: Dict, control_data: Dict, flt) -> Dict:
        if flt == "all":
            return patient_data

        source = flt.get("source", "patient")
        data = patient_data if source == "patient" else control_data
        filter_type = flt.get("filter", "all")

        if filter_type == "all":
            return data

        if filter_type == "subject_prefix":
            prefix = flt["prefix"]
            return {k: v for k, v in data.items()
                    if extract_subject_id(k).startswith(prefix)}

        if filter_type == "metadata":
            ids = self._get_metadata_ids(flt["column"], flt["values"])
            return {k: v for k, v in data.items()
                    if extract_subject_id(k) in ids}

        if filter_type == "metadata_exclude":
            ids = self._get_metadata_ids(flt["column"], flt["values"])
            return {k: v for k, v in data.items()
                    if extract_subject_id(k) not in ids}

        raise ValueError(f"Unknown filter type: {filter_type!r}")
