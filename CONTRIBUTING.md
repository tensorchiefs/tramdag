# Contributing

## Setup

```bash
uv sync                 # creates .venv from the pinned uv.lock
```

With [direnv](https://direnv.net/), `.envrc` activates `.venv` on `cd`.

## Tests

```bash
uv run pytest -q -m "not slow"          # the fast subset
uv run pytest -q                        # everything, incl. the long fits
uv run pytest tests/test_flow.py -q     # one file
```

Run `pytest` with no path: `testpaths` in `pyproject.toml` then picks up both
`tests/` and the per-area `experiments/*/tests/`.

What the suite guarantees, what `slow` marks, how CI splits the runs, and
where each reference number comes from is documented in
[`tests/README.md`](tests/README.md). Two rules matter most:

- `data/` is a **contract**. A new seed or changed equations means a new
  folder, never an edit in place.
- Validate a new causal feature against a simulator's known truth, not
  just "it runs".

## Linting

```bash
uvx ruff check .        # report
uvx ruff format --diff  # what formatting would change
```

Rules live in `pyproject.toml`: ruff's default set plus the extras listed
under `extend-select`, at 88 columns with the numpy docstring convention.
The complexipy hooks gate cognitive complexity. The limit is 15 for `src/`
and `tools/`. The limit is 10 for `experiments/`, `notebooks/` and `tests/`.
Docstrings are not required in `tests/`, `experiments/` or `notebooks/`.

The hooks in `.pre-commit-config.yaml` are enforced by
`.github/workflows/pre-commit.yaml` on every push and pull request. Install
them locally so a push cannot fail on formatting:

```bash
pre-commit install --install-hooks -t pre-commit -t commit-msg
```

## Module layout

Every Python module reads in the same order, separated by `# %%` markers
padded with dashes to column 88:

```python
# %% imports ---------------------------------------------------------------------------
# %% global variables ------------------------------------------------------------------
# %% private functions -----------------------------------------------------------------
# %% public functions ------------------------------------------------------------------
# %% private classes -------------------------------------------------------------------
# %% public classes --------------------------------------------------------------------
# %% alias -----------------------------------------------------------------------------
# %% main ------------------------------------------------------------------------------
```

A module carries only the sections it has. These eight are the only section
names. There are no sub-section banners of any other kind. Notebooks are
jupytext `py:percent` files, and they keep their narrative cell structure
instead.

## Notebooks

`notebooks/*.py` are [jupytext](https://github.com/mwouts/jupytext)
percent-format files and are the source of truth. Edit those, never an
`.ipynb`. One `.ipynb` is tracked on purpose. It is
`demo_tram_dag_colab.ipynb`, because the README's Colab badge links to it.
Regenerate it from the `.py`. See
[`notebooks/README.md`](notebooks/README.md).

## Conventions worth knowing

Four implementation conventions are easy to get wrong:

- the latent-scale signs,
- the raw parent encoding against the one-hot parent encoding,
- the log-space ordinal likelihood,
- the seeding.

[`CLAUDE.md`](CLAUDE.md) documents all four, and tests pin them. Read that
document before you change anything in `src/tramdag/`.
