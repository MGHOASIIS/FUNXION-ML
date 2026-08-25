# paper/

Code that exists to reproduce a specific figure or table in a specific
published/submitted paper — not reusable pipeline tooling.

Scripts here are expected to:
- Target one exact figure/table, not a general task/paradigm/model combination.
- Often hardcode numbers copied from a specific results run rather than
  reading a checkpoint live, and use paper-specific formatting (fonts, colors,
  exact output filenames).
- Not take `--dataset`/`--model` style arguments — if a script would be
  useful for any dataset or task, it belongs in `scripts/` instead, not here.

Generated outputs (figures, tables, results, ...) go under `storage/paper/{area}/`
(gitignored, mirroring the `paper/{area}/` scripts that produced them) —
never commit generated figures alongside the script. Use
`config.paths.get_paper_dir(area)` to get that path rather than writing to
the current working directory.

If a script computes something (not just plots pre-computed numbers), prefer
having it read from `storage/results/{dataset}/...` rather than hardcoding
values, so it stays connected to its source of truth — see
`scripts/hmm/save_emission_importance.py` for the pattern to follow when a
paper figure needs fresh numbers instead of a copy-pasted snapshot.

## Layout

One subfolder per model/analysis area, matching the `scripts/` convention
(`scripts/hmm/`, `scripts/nn/`, ...) — group by what the figure is about, not
by figure number.

- `paper/hmm/` — figures/tables derived from HMM or HSMM results, e.g.
  `make_permutation_vs_emission_figures.py`.
