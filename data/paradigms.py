"""
Data paradigm selection for classification tasks.

Paradigms:
1. Patients vs Controls
2. Rotator Cuff Tear (RCT) vs Controls
3. Other Conditions vs Controls
4. RCT vs Other Conditions

Works with both data formats:
  - Legacy:       key = subject_id         (e.g. "PX01", "fx01")
  - Event-window: key = window_id          (e.g. "fx07_task4_trial1")
Subject identity is always extracted via extract_subject_id(key).
"""
from typing import Dict, Tuple
import pandas as pd
from config.paths import PATIENT_DETAILS
from config.constants import EXCLUDED_G1, EXCLUDED_G0


def extract_subject_id(key: str) -> str:
    """
    Extract the subject identifier from a dict key.

    Handles both formats:
      - subject_id key : "PX01"              -> "PX01"
      - window_id key  : "fx07_task4_trial1" -> "fx07"
    """
    return key.split('_')[0]


class ParadigmSelector:
    """Handles data filtering for different classification paradigms."""
    
    def __init__(self):
        """Load patient diagnosis information."""
        self.df_px_info = self._load_patient_details()
        self.rotator_cuff_ids = self.df_px_info[
            self.df_px_info['dia_code'] == 1
        ]['id'].tolist()

    @staticmethod
    def _load_patient_details() -> pd.DataFrame:
        """
        Load xdash_px_details.xlsx, auto-detecting the correct header row.

        The xlsx has a notes row above the actual column headers, so the
        header row index can shift if the file has been modified (e.g. after
        adding survey response columns with openpyxl). Instead of hardcoding
        header=1, we probe rows 0–5 until we find the one that contains
        'dia_code' as a column name.
        """
        for header_row in range(6):
            df = pd.read_excel(
                PATIENT_DETAILS,
                sheet_name='Sheet1',
                header=header_row
            )
            if 'dia_code' in df.columns:
                if header_row != 1:
                    print(f"[ParadigmSelector] Note: found 'dia_code' at header={header_row} "
                          f"(expected 1) — xlsx may have been modified.")
                return df

        # Last resort: return with header=1 and let the caller raise a clear error
        raise ValueError(
            f"Could not find 'dia_code' column in any of the first 6 rows of "
            f"{PATIENT_DETAILS}.\n"
            f"Please check that the file is the correct xdash_px_details.xlsx."
        )
    

    def select_paradigm(
        self,
        patient_data: Dict,
        control_data: Dict,
        paradigm: int
    ) -> Tuple[Dict, Dict]:
        """
        Select data groups based on paradigm.
        
        Parameters
        ----------
        patient_data : Dict
            Dictionary of patient data {id: tensor}
        control_data : Dict
            Dictionary of control data {id: tensor}
        paradigm : int
            Classification paradigm (1-4)
        
        Returns
        -------
        g1, g0 : Tuple[Dict, Dict]
            Data dictionaries for group 1 and group 0
        """
        # Apply global exclusions before any paradigm filtering
        patient_data = {k: v for k, v in patient_data.items() if extract_subject_id(k) not in self.EXCLUDE_G1}
        control_data = {k: v for k, v in control_data.items() if extract_subject_id(k) not in self.EXCLUDE_G0}

        excluded_g1 = [k for k in self.EXCLUDE_G1]
        excluded_g0 = [k for k in self.EXCLUDE_G0]
        print(f"[ParadigmSelector] Excluded from g1: {excluded_g1}")
        print(f"[ParadigmSelector] Excluded from g0: {excluded_g0}")
        if paradigm == 1:
            return self._patients_vs_controls(patient_data, control_data)
        elif paradigm == 2:
            return self._rct_vs_controls(patient_data, control_data)
        elif paradigm == 3:
            return self._other_vs_controls(patient_data, control_data)
        elif paradigm == 4:
            return self._rct_vs_other(patient_data)
        else:
            raise ValueError(f"Unknown paradigm: {paradigm}")
    
    def _patients_vs_controls(
        self,
        patient_data: Dict,
        control_data: Dict
    ) -> Tuple[Dict, Dict]:
        """Paradigm 1: All patients vs all controls."""
        print(f"Paradigm 1: Patients vs Controls")
        print(f"  Group 1 (Patients): {len(patient_data)}")
        print(f"  Group 0 (Controls): {len(control_data)}")
        return patient_data, control_data
    
    def _rct_vs_controls(
        self,
        patient_data: Dict,
        control_data: Dict
    ) -> Tuple[Dict, Dict]:
        """Paradigm 2: RCT patients vs controls."""
        rct_data = {
            k: v for k, v in patient_data.items()
            if extract_subject_id(k) in self.rotator_cuff_ids
        }

        filtered_controls = {
            k: v for k, v in control_data.items()
            if extract_subject_id(k).startswith('fx')
        }
        
        print(f"Paradigm 2: RCT vs Controls")
        print(f"  Group 1 (RCT): {len(rct_data)}")
        print(f"  Group 0 (Controls): {len(filtered_controls)}")
        return rct_data, filtered_controls
    
    def _other_vs_controls(
        self,
        patient_data: Dict,
        control_data: Dict
    ) -> Tuple[Dict, Dict]:
        """Paradigm 3: Non-RCT patients vs controls."""
        other_data = {
            k: v for k, v in patient_data.items()
            if extract_subject_id(k) not in self.rotator_cuff_ids
        }

        filtered_controls = {
            k: v for k, v in control_data.items()
            if extract_subject_id(k).startswith('fx')
        }
        
        print(f"Paradigm 3: Other Conditions vs Controls")
        print(f"  Group 1 (Other): {len(other_data)}")
        print(f"  Group 0 (Controls): {len(filtered_controls)}")
        return other_data, filtered_controls
    
    def _rct_vs_other(
        self,
        patient_data: Dict
    ) -> Tuple[Dict, Dict]:
        """Paradigm 4: RCT vs other conditions."""
        rct_data = {
            k: v for k, v in patient_data.items()
            if extract_subject_id(k) in self.rotator_cuff_ids
        }
        other_data = {
            k: v for k, v in patient_data.items()
            if extract_subject_id(k) not in self.rotator_cuff_ids
        }
        
        print(f"Paradigm 4: RCT vs Other Conditions")
        print(f"  Group 1 (RCT): {len(rct_data)}")
        print(f"  Group 0 (Other): {len(other_data)}")
        return rct_data, other_data