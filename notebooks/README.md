# Notebooks

All notebooks in this repo are [jupytext](https://github.com/mwouts/jupytext)
**percent-format `.py` files** — plain Python with `# %%` cell markers and
markdown in `# %% [markdown]` cells. The `.py` file is always the **source of
truth**.

| notebook | what it is |
|---|---|
| `demo_tram_dag_colab.py` | the introduction and the showcase in one: the paper's bimodal VACA benchmark, L1 to L3, then the same DAG written with interpretable terms ([open in Colab](https://colab.research.google.com/github/tensorchiefs/tramdag/blob/main/notebooks/demo_tram_dag_colab.ipynb)) |
| `training_strategies.py` | every shipped fitting recipe on one workload, from the API side: `fit`'s own records, plain Adam, two phases, `EarlyStopping`, `PerNodePlateau`, and a `Callback` of your own |
| `additive_vs_joint_ci.py` | joint vs additive complex intercept, and reading per-parent effects out of the additive one with `intercept_contributions` |
| `varying_coefficients.py` | heterogeneous treatment effects: the `VC` head, `varying_coef`, the modifier scan and propensity centering, all scored against a known `beta(x)` |
| `classical_fit_tram_dag.py` | `fit_classical` on all-`ls` models, opening with plain logistic regression on `MASS::birthwt` (a 2-level ordinal node) checked against R `glm`: determinism, the exact MLE against `statsmodels` / R, and the classical-fit-then-keep-training warm start |

`classical_fit_tram_dag.R` is not a notebook. It is the R half of
`classical_fit_tram_dag.py`. It fits every classical reference that the
notebook holds as a constant, in one script, so the numbers can be re-checked
instead of trusted. It needs `tram`, which CI does not install, so run it by
hand:
`Rscript notebooks/classical_fit_tram_dag.R` from the repo root.

The docs workflow executes every notebook here, on pushes to `main` and to
`dev-*` branches. That execution is what keeps them working against the
current API. A notebook that is not in that workflow's loop does not belong in
this directory. On a feature branch, run one by hand:
`MPLBACKEND=Agg uv run python notebooks/<name>.py`.

Data a notebook reads lives in `notebooks/data/` — see its README for
provenance.

## Rules

- **Do not edit `.ipynb` files directly** — edit the `.py` and regenerate.
- **Do not commit `.ipynb` files.** They are git-ignored (embedded base64
  outputs ruin diffs). The single exception is `demo_tram_dag_colab.ipynb`,
  tracked output-stripped, only so that the Open-in-Colab badge works. If you
  change `demo_tram_dag_colab.py`, regenerate the `.ipynb` before you commit.

## Working with the notebooks

**VS Code / Cursor**: open the `.py` directly. With the Python and Jupyter
extensions, every `# %%` cell gets a "Run Cell" link. Pick the `.venv`
interpreter that `uv sync` created. No conversion is needed.

**Classic Jupyter / JupyterLab**: generate a local `.ipynb` (stays untracked):

```bash
uvx jupytext --to ipynb notebooks/training_strategies.py
```

For frequent notebook editing you can install jupytext into the venv instead of
using `uvx` each time: `uv sync --group notebooks`.

**Edit in a synced copy.** The `.py` stays the source of truth. With the
`notebooks` group installed, jupytext keeps a local `.ipynb` paired to the
`.py`. Your interactive edits then flow back into the tracked `.py`.

The cleanest way needs no `.ipynb` at all — in JupyterLab/Jupyter Notebook,
right-click the `.py` → *Open With* → *Notebook*. Edits save straight back to the
`.py`, and there is nothing to clean up. If you prefer a real paired
`.ipynb`, jupytext can do that too. Use `--set-formats ipynb,py:percent` and
then `--sync`.

The paired `.ipynb` stays git-ignored. Note that `--set-formats` adds `ipynb` to
the `.py` header — revert that one-line header change before committing (the
committed notebooks are paired to `py:percent` only).

**Headless check** (runs all cells top-to-bottom, plots suppressed):

```bash
MPLBACKEND=Agg uv run python notebooks/training_strategies.py
```

**Regenerate the tracked Colab demo ipynb** after editing its `.py`:

```bash
uvx jupytext --to ipynb notebooks/demo_tram_dag_colab.py
```

(A fresh conversion contains no outputs, which is exactly the committed state.)

More on the format: [jupytext documentation](https://jupytext.readthedocs.io).
