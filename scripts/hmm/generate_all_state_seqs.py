"""
generate_all_state_seqs.py
================================
Generates HMM state sequence plots for every subject across all
task × paradigm combinations (T1–T6 × P1–P4 = up to 24 experiments).

Output structure:
    hmm-results/
      state_seqs/
        T1_P1/   (jar_opening | patients_vs_controls)
          patients/    ← one PNG per subject in group 1
          controls/    ← one PNG per subject in group 0
          summary.csv
        T1_P2/   (jar_opening | rct_vs_controls)
          rct/
          controls/
          summary.csv
        ...
        T6_P4/   (hammering | rct_vs_other)
          rct/
          other/
          summary.csv

Checkpoint discovery:
    Looks for:  hmm-results/T{t}-P{p}-HMM*/model_checkpoints/HMM_T{t}_P{p}_BA*.json
    Falls back: hmm-results/*/HMM_T{t}_P{p}_BA*.json (recursive)
    Skips T/P combinations where no checkpoint is found.

Usage (from project root):
    # All tasks and paradigms
    python scripts/generate_all_state_seqs.py

    # Specific task only
    python scripts/generate_all_state_seqs.py --tasks 1 2

    # Specific paradigm only
    python scripts/generate_all_state_seqs.py --paradigms 1

    # Custom checkpoint root
    python scripts/generate_all_state_seqs.py --hmm-dir hmm-results/

    # With event CSV overlay
    python scripts/generate_all_state_seqs.py --csv-dir data/events/
"""
import argparse
import glob
import json
import pickle
import sys
from pathlib import Path

import numpy as np
import pandas as pd


# ── Paradigm metadata ─────────────────────────────────────────────────────────
# Maps paradigm → (group1_folder_name, group0_folder_name)
PARADIGM_GROUP_NAMES = {
    1: ("patients",  "controls"),
    2: ("rct",       "controls"),
    3: ("other",     "controls"),
    4: ("rct",       "other"),
}

TASK_NAMES = {
    1: "jar_opening",
    2: "key_turning",
    3: "cleaning",
    4: "back_washing",
    5: "cutting",
    6: "hammering",
}

PARADIGM_NAMES = {
    1: "patients_vs_controls",
    2: "rct_vs_controls",
    3: "other_conditions_vs_controls",
    4: "rct_vs_other_conditions",
}


# ── CLI ───────────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser(
        description="Batch HMM state-sequence plots for all T×P combinations"
    )
    p.add_argument("--tasks",      nargs="+", type=int, default=list(range(1, 7)),
                   help="Task numbers to process (default: 1-6)")
    p.add_argument("--paradigms",  nargs="+", type=int, default=list(range(1, 5)),
                   help="Paradigm numbers to process (default: 1-4)")
    p.add_argument("--hmm-dir",    default="experiments_from_hpc/",
                   help="Root directory containing HMM experiment folders (default: hmm-results)")
    p.add_argument("--out-dir",    default="hmm-results/state_seqs",
                   help="Output root (default: hmm-results/state_seqs)")
    p.add_argument("--csv-dir",    default=None,
                   help="Directory with consolidated_task{n}.csv event files (e.g. data/events/)")
    p.add_argument("--sampling-rate", type=int, default=50,
                   help="Sampling rate in Hz (default: 50)")
    p.add_argument("--skip-existing", action="store_true",
                   help="Skip T/P combinations whose output folder already exists")
    return p.parse_args()


# ── Project import ────────────────────────────────────────────────────────────

def setup_project_path() -> bool:
    """Add project root to sys.path. Returns True on success."""
    script_dir = Path(__file__).resolve().parent
    candidates = [Path.cwd(), script_dir, script_dir.parent]
    for root in candidates:
        if (root / "models" / "hmm_model.py").exists():
            sys.path.insert(0, str(root))
            try:
                import models.hmm_model  # noqa: F401
                print(f"[Import] Project root: {root}")
                return True
            except ImportError:
                sys.path.pop(0)
    return False


# ── Checkpoint discovery ──────────────────────────────────────────────────────

def find_checkpoint(hmm_dir: Path, task: int, paradigm: int) -> Path | None:
    """
    Search for the best HMM checkpoint for a given T/P combination.

    Search order:
      1. hmm-results/T{t}-P{p}-HMM*/model_checkpoints/HMM_T{t}_P{p}_BA*.json
      2. hmm-results/**/HMM_T{t}_P{p}_BA*.json  (recursive fallback)
    """
    # Primary: standard experiment folder layout
    patterns = [
                str(hmm_dir/f"task{task}"/f"paradigm{paradigm}"/"HMM*"/"model_checkpoints"/f"results_T{task}_P{paradigm}_BA*.json"),
                str(hmm_dir / f"HMM_T{task}_P{paradigm}" / "model_checkpoints"/f"HMM_T{task}_P{paradigm}_BA*.json"),
        # Recursive fallback
        str(hmm_dir / "**" / f"HMM_T{task}_P{paradigm}_BA*.json"),
    ]

    for pattern in patterns:
        matches = sorted(glob.glob(pattern, recursive=True))
        if matches:
            if len(matches) > 1:
                print(f"  [WARN] Multiple checkpoints found — using highest BA:")
                # Sort by BA value extracted from filename
                def extract_ba(p):
                    try:
                        return float(Path(p).stem.split("_BA")[1].split("_")[0])
                    except Exception:
                        return 0.0
                matches = sorted(matches, key=extract_ba, reverse=True)
                print(f"    → {matches[0]}")
            return Path(matches[0])

    return None


# ── Data loading ──────────────────────────────────────────────────────────────

_data_cache: dict = {}  # task → (patient_data, control_data)


def load_task_data(task: int) -> tuple:
    """Load and cache patient/control data for a task."""
    if task in _data_cache:
        return _data_cache[task]

    for base in [Path("data/pickled_datasets"), Path("data")]:
        p_path = base / f"patient_data_task{task}.pkl"
        c_path = base / f"control_data_task{task}.pkl"
        if p_path.exists() and c_path.exists():
            with open(p_path, "rb") as f:
                patients = pickle.load(f)
            with open(c_path, "rb") as f:
                controls = pickle.load(f)
            print(f"  [Data] task={task}: {len(patients)} patients, {len(controls)} controls")
            _data_cache[task] = (patients, controls)
            return patients, controls

    raise FileNotFoundError(
        f"Pickled data not found for task {task}. "
        "Expected: data/pickled_datasets/patient_data_task{task}.pkl"
    )


# ── Preprocessing ─────────────────────────────────────────────────────────────

def preprocess(g1: dict, g0: dict):
    """
    VariableLengthPreprocessor logic:
      - Extract 18-channel signals (strip timestamp col if present)
      - Fit StandardScaler on all frames concatenated
      - Return X (list of arrays), y, subject_ids
    """
    import torch
    from sklearn.preprocessing import StandardScaler

    all_tensors, labels, sids = [], [], []
    for sid, t in g1.items():
        all_tensors.append(t); labels.append(1); sids.append(sid)
    for sid, t in g0.items():
        all_tensors.append(t); labels.append(0); sids.append(sid)

    signals = []
    for t in all_tensors:
        arr = t.detach().cpu().numpy() if isinstance(t, torch.Tensor) else np.asarray(t)
        signals.append(arr[:, 1:] if arr.shape[1] > 18 else arr)

    scaler = StandardScaler()
    scaler.fit(np.vstack(signals))
    X = [scaler.transform(s) for s in signals]
    return X, np.array(labels), sids


# ── Per T/P runner ────────────────────────────────────────────────────────────

def run_one(task: int, paradigm: int, ckpt_path: Path,
            out_root: Path, csv_dir: Path | None,
            sampling_rate: int) -> dict:
    """
    Generate state sequence plots for one task/paradigm combination.
    Returns a dict with summary statistics.
    """
    from models.hmm_model import HMMModel
    from data.paradigms import ParadigmSelector

    tag       = f"T{task}_P{paradigm}"
    task_name = TASK_NAMES.get(task, f"task{task}")
    p_name    = PARADIGM_NAMES.get(paradigm, f"paradigm{paradigm}")
    g1_name, g0_name = PARADIGM_GROUP_NAMES[paradigm]

    # Output directories
    out_dir = out_root / tag
    (out_dir / g1_name).mkdir(parents=True, exist_ok=True)
    (out_dir / g0_name).mkdir(parents=True, exist_ok=True)

    print(f"\n{'='*65}")
    print(f"  {tag}  |  {task_name}  |  {p_name}")
    print(f"  Checkpoint: {ckpt_path.name}")
    print(f"  Output:     {out_dir}")
    print(f"{'='*65}")

    # Load checkpoint
    with open(ckpt_path) as f:
        ckpt = json.load(f)
    hp = ckpt["hyperparameters"]
    print(f"  BA={ckpt['metrics'].get('balanced_accuracy', 'N/A')}  "
          f"params={hp}")

    # Load raw data
    patient_data, control_data = load_task_data(task)

    # Apply paradigm filter
    selector = ParadigmSelector()
    g1, g0 = selector.select_paradigm(patient_data, control_data, paradigm)

    # Preprocess
    X, y, sids = preprocess(g1, g0)
    print(f"  Subjects: {len(X)}  "
          f"(g1={int(y.sum())} {g1_name}, g0={int((y==0).sum())} {g0_name})")

    # Fit class-conditional HMMs on full data
    hmm = HMMModel(task=task, paradigm=paradigm)
    hmm.fit_for_analysis(
        X=X, y=y,
        n_components=hp["n_components"],
        covariance_type=hp["covariance_type"],
        n_iter=hp["n_iter"],
    )

    # Event CSV for this task
    csv_path = None
    if csv_dir:
        candidate = Path(csv_dir) / f"consolidated_task{task}.csv"
        if candidate.exists():
            csv_path = candidate
        else:
            print(f"  [WARN] Event CSV not found: {candidate}")

    # Decode and plot every subject
    summary_rows = []

    for i, (seq, label, sid) in enumerate(zip(X, y, sids)):
        is_g1  = (label == 1)
        model  = hmm.fitted_hmm1 if is_g1 else hmm.fitted_hmm0
        subdir = out_dir / (g1_name if is_g1 else g0_name)
        group  = g1_name if is_g1 else g0_name

        # Strip preprocessor prefix (g1_0_PX01 → PX01)
        clean_sid = sid.split("_", 2)[-1] if sid.startswith(("g1_", "g0_")) else sid

        print(f"  [{i+1:2d}/{len(X)}] {clean_sid:8s} ({group})", end=" ")

        # Decode
        states, _, _ = hmm.decode_sequence(seq, model=model,
                                            sampling_rate=sampling_rate)

        # Events
        events = None
        if csv_path:
            try:
                events = hmm.load_event_markers(
                    csv_path=csv_path,
                    subject_id=clean_sid,
                    task_id=task,
                    relative_timestamps=True,
                )
            except Exception:
                pass

        # Plot
        save_path = subdir / f"state_seq_{clean_sid}.png"
        hmm.plot_state_sequence_over_time(
            sequence=seq,
            state_sequence=states,
            events=events,
            sampling_rate=sampling_rate,
            title=f"State Seq — {clean_sid} ({group}) {tag}",
            save_path=save_path,
        )

        # Stats — dynamically handle any n_components (2, 3, 4 ...)
        n_trans = sum(1 for j in range(1, len(states)) if states[j] != states[j-1])
        total_s = len(states) / sampling_rate

        row = {
            "task":          task,
            "paradigm":      paradigm,
            "subject_id":    clean_sid,
            "group":         group,
            "n_frames":      len(states),
            "total_s":       round(total_s, 1),
            "n_transitions": n_trans,
        }
        # Write stateN_pct for every state 0 .. n_components-1
        n_components = hp["n_components"]
        state_pct_str = []
        for s_idx in range(n_components):
            pct = round((states == s_idx).mean() * 100, 1)
            row[f"state{s_idx}_pct"] = pct
            state_pct_str.append(f"S{s_idx}={pct:.0f}%")

        summary_rows.append(row)
        print(f"✓  {total_s:.0f}s | {n_trans} trans | {' '.join(state_pct_str)}")

    # Summary CSV
    df = pd.DataFrame(summary_rows)
    df_sorted = pd.concat([
        df[df["group"] == g1_name].sort_values("subject_id"),
        df[df["group"] == g0_name].sort_values("subject_id"),
    ])
    df_sorted.to_csv(out_dir / "summary.csv", index=False)

    # Group stats
    # Build stats with all available stateN_pct columns
    state_pct_cols = sorted([c for c in df.columns if c.startswith("state") and c.endswith("_pct")])
    stats = df.groupby("group")[["total_s", "n_transitions"] + state_pct_cols].mean().round(1)
    print(f"\n  Group means:\n{stats.to_string()}")

    return {
        "tag":      tag,
        "task":     task,
        "paradigm": paradigm,
        "n_total":  len(summary_rows),
        "n_g1":     int(y.sum()),
        "n_g0":     int((y == 0).sum()),
        "out_dir":  str(out_dir),
    }


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    args = parse_args()

    # Project imports
    if not setup_project_path():
        print("[ERROR] Could not locate project root (models/hmm_model.py not found).")
        print("Run from your project root directory.")
        sys.exit(1)

    hmm_dir  = Path(args.hmm_dir)
    out_root = Path(args.out_dir)
    csv_dir  = Path(args.csv_dir) if args.csv_dir else None

    out_root.mkdir(parents=True, exist_ok=True)
    print(f"\n[Output root] {out_root.resolve()}")
    print(f"[HMM dir]     {hmm_dir.resolve()}")

    # Build list of (task, paradigm) combinations to process
    combos = [(t, p) for t in args.tasks for p in args.paradigms]
    print(f"\n[Plan] {len(combos)} task×paradigm combinations: "
          f"tasks={args.tasks}  paradigms={args.paradigms}\n")

    results   = []
    skipped   = []
    no_ckpt   = []

    for task, paradigm in combos:
        tag = f"T{task}_P{paradigm}"

        # Skip if output already exists
        if args.skip_existing and (out_root / tag / "summary.csv").exists():
            print(f"[SKIP] {tag} — already done")
            skipped.append(tag)
            continue

        # Find checkpoint
        ckpt_path = find_checkpoint(hmm_dir, task, paradigm)
        if ckpt_path is None:
            print(f"[SKIP] {tag} — no checkpoint found in {hmm_dir}")
            no_ckpt.append(tag)
            continue

        try:
            result = run_one(
                task=task,
                paradigm=paradigm,
                ckpt_path=ckpt_path,
                out_root=out_root,
                csv_dir=csv_dir,
                sampling_rate=args.sampling_rate,
            )
            results.append(result)
        except Exception as e:
            import traceback
            print(f"\n[ERROR] {tag} failed: {e}")
            traceback.print_exc()
            skipped.append(tag)

    # ── Final summary ─────────────────────────────────────────────────────────
    print(f"\n{'='*65}")
    print("BATCH COMPLETE")
    print(f"{'='*65}")
    print(f"  Processed : {len(results)}")
    print(f"  No ckpt   : {len(no_ckpt)}  {no_ckpt}")
    print(f"  Errors    : {len(skipped)}  {skipped}")
    print(f"\nOutput layout:")
    print(f"  {out_root}/")
    for r in sorted(results, key=lambda x: (x["task"], x["paradigm"])):
        g1n, g0n = PARADIGM_GROUP_NAMES[r["paradigm"]]
        print(f"    {r['tag']}/  "
              f"({TASK_NAMES[r['task']]} | {PARADIGM_NAMES[r['paradigm']]})")
        print(f"      {g1n}/  ({r['n_g1']} subjects)")
        print(f"      {g0n}/  ({r['n_g0']} subjects)")
        print(f"      summary.csv")

    # Master summary across all T/P
    if results:
        all_summaries = []
        for r in results:
            csv_path = Path(r["out_dir"]) / "summary.csv"
            if csv_path.exists():
                all_summaries.append(pd.read_csv(csv_path))
        if all_summaries:
            master = pd.concat(all_summaries, ignore_index=True)
            master_path = out_root / "master_summary.csv"
            master.to_csv(master_path, index=False)
            print(f"\n  Master summary: {master_path}  ({len(master)} rows)")

    print(f"{'='*65}\n")


if __name__ == "__main__":
    main()