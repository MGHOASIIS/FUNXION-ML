"""
Data paradigm selection for classification tasks.

Paradigms:
1. Patients vs Controls
2. Rotator Cuff Tear (RCT) vs Controls
3. Other Conditions vs Controls
4. RCT vs Other Conditions
"""
from typing import Dict, Tuple
import pandas as pd
from config.paths import PATIENT_DETAILS


class ParadigmSelector:
    """Handles data filtering for different classification paradigms."""
    
    def __init__(self):
        """Load patient diagnosis information."""
        self.df_px_info = pd.read_excel(PATIENT_DETAILS, sheet_name='Sheet1', header=1)
        self.rotator_cuff_ids = self.df_px_info[
            self.df_px_info['dia_code'] == 1
        ]['id'].tolist()
    
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
            if k in self.rotator_cuff_ids
        }
        
        # Filter controls to match RCT patients if needed
        filtered_controls = {
            k: v for k, v in control_data.items()
            if (k.startswith('PX') and k in self.rotator_cuff_ids) or k.startswith('fx')
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
            if k not in self.rotator_cuff_ids
        }
        
        # Filter controls
        filtered_controls = {
            k: v for k, v in control_data.items()
            if (k.startswith('PX') and k in self.rotator_cuff_ids) or k.startswith('fx')
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
            if k in self.rotator_cuff_ids
        }
        other_data = {
            k: v for k, v in patient_data.items()
            if k not in self.rotator_cuff_ids
        }
        
        print(f"Paradigm 4: RCT vs Other Conditions")
        print(f"  Group 1 (RCT): {len(rct_data)}")
        print(f"  Group 0 (Other): {len(other_data)}")
        return rct_data, other_data