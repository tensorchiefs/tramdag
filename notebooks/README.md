# Notebooks

All notebooks in this repo are [jupytext](https://github.com/mwouts/jupytext)
**percent-format `.py` files** — plain Python with `# %%` cell markers and
markdown in `# %% [markdown]` cells. The `.py` file is always the **source of
truth**.

| notebook | what it is |
|---|---|
| `intro_tram_dag.py` | didactic walkthrough of the TRAM-DAG model (SI/CI/LS/CS, L1–L3, all claims checked against a hand-built SCM) |
| `demo_tram_dag_colab.py` | 5-minute showcase on the paper's bimodal VACA benchmark, GPU-ready ([open in Colab](https://colab.research.google.com/github/tensorchiefs/tramdag/blob/main/notebooks/demo_tram_dag_colab.ipynb)) |

## Rules

- **Do not edit `.ipynb` files directly** — edit the `.py` and regenerate.
- **Do not commit `.ipynb` files.** They are git-ignored (embedded base64
  outputs ruin diffs). The single exception is `demo_tram_dag_colab.ipynb`,
  tracked **output-stripped** only so the Open-in-Colab badge works; after
  changing `demo_tram_dag_colab.py`, regenerate it before committing.

## Working with the notebooks

**VS Code / Cursor**: open the `.py` directly — with the Python + Jupyter
extensions every `# %%` cell gets a "Run Cell" link (pick the `.venv`
interpreter created by `uv sync`). No conversion needed.

**Classic Jupyter / JupyterLab**: generate a local `.ipynb` (stays untracked):

```bash
uvx jupytext --to ipynb notebooks/intro_tram_dag.py
```

**Headless check** (runs all cells top-to-bottom, plots suppressed):

```bash
MPLBACKEND=Agg uv run python notebooks/intro_tram_dag.py
```

**Regenerate the tracked Colab demo ipynb** after editing its `.py`:

```bash
uvx jupytext --to ipynb notebooks/demo_tram_dag_colab.py
```

(A fresh conversion contains no outputs, which is exactly the committed state.)

More on the format: [jupytext documentation](https://jupytext.readthedocs.io).
