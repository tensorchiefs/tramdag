# Notebooks

The notebooks are [jupytext](https://jupytext.readthedocs.io) percent-format
`.py` files: plain Python with `# %%` cell markers and markdown in
`# %% [markdown]` cells. The `.py` file is the source of truth.

| notebook | what it is |
|---|---|
| `demo_tram_dag_colab.py` | the introduction: the paper's bimodal VACA benchmark, L1 to L3, each checked against the known truth ([open in Colab](https://colab.research.google.com/github/tensorchiefs/tramdag/blob/main/notebooks/demo_tram_dag_colab.ipynb)) |
| `training_strategies.py` | every shipped fitting recipe on one workload: `fit`'s records, plain Adam, two phases, `EarlyStopping`, per-node stops, a custom `Callback`, an optimizer factory, one node alone, and the runtimes |
| `fit_strategies_benchmark.py` | `fit` per node, in series and in processes, against `fit_jointly`, each with and without a plateau schedule: runtime, convergence, validation NLL and an intervention error |
| `additive_vs_joint_ci.py` | joint against additive complex intercept, and per-parent effects from the additive one with `intercept_contributions` |
| `varying_coefficients.py` | heterogeneous treatment effects: the `VC` head, `varying_coef`, the modifier scan and propensity centering, all scored against a known `beta(x)` |
| `classical_fit_tram_dag.py` | `fit_classical` on all-`LS` models against `statsmodels` and R: logistic regression on `MASS::birthwt`, the same regression as a lone `Node`, and the warm start |

`classical_fit_tram_dag.R` is the R half of `classical_fit_tram_dag.py`. It
fits every R reference that the notebook holds as a constant. It needs
`tram`, which CI does not install, so run it by hand from the repository
root: `Rscript notebooks/classical_fit_tram_dag.R`.

The docs workflow executes every notebook on pushes to `main` and to `dev-*`
branches. A notebook that is not in that workflow does not belong here.
`notebooks/data/README.md` gives the provenance of the data that the
notebooks read.

## Rules

- Do not edit `.ipynb` files. Edit the `.py` and regenerate.
- Do not commit `.ipynb` files. Git ignores them. The exception is
  `demo_tram_dag_colab.ipynb`, tracked without outputs for the Colab badge.
  Regenerate it when its `.py` changes.

## Working with the notebooks

In VS Code, open the `.py`: every `# %%` cell gets a "Run Cell" link. In
JupyterLab, right-click the `.py` and select *Open With*, then *Notebook*.
The edits save to the `.py`.

Run a notebook headless from the repository root:

```bash
MPLBACKEND=Agg uv run python notebooks/training_strategies.py
```

Regenerate the tracked Colab notebook after you edit its `.py`. `--update`
keeps the cell identifiers, so the diff shows only the changed cells:

```bash
uvx jupytext --update --to ipynb notebooks/demo_tram_dag_colab.py
```
