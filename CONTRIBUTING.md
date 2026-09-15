# Contributing

## Setup

```bash
uv sync                 # creates .venv from the pinned uv.lock
```

With [direnv](https://direnv.net/), `.envrc` activates `.venv` on `cd`.

## Tests

```bash
uv run pytest -q -m "not slow"          # the fast subset, 2-3 min
uv run pytest -q                        # everything, incl. the long fits (25-40 min on 2-core CI)
uv run pytest tests/test_flow.py -q     # one file
```

Run `pytest` with no path: `testpaths` in `pyproject.toml` then picks up both
`tests/` and the per-area `experiments/*/tests/`.

What the suite guarantees, what `slow` marks, how CI splits the runs, and
where each reference number comes from is documented in
[`tests/README.md`](tests/README.md). The policy:

- Validate a new causal feature against the known truth of an inline DGP in
  `tests/conftest.py`, never against "runs without error". If no DGP fits,
  add one.
- Mark a long fit `@pytest.mark.slow` so PR CI stays fast, but never the fit
  that *is* a feature's acceptance measurement.
- A framework test never imports `experiments/`. The research generators and
  their frozen CSVs live there, and the experiments workflow checks them.
- `experiments/<area>/data/` is a contract. A new seed or changed equations
  means a new folder, never an edit in place.

## Linting

```bash
uvx ruff check .                    # report
uvx ruff format --diff src tests experiments tools notebooks   # formatting
```

`ruff format` also reformats fenced Python inside markdown, which the
`ruff-format` hook deliberately does not do. Code in the guides is wrapped for
reading, not for the formatter, so point the command at the source trees.

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

Four implementation conventions are easy to get wrong, and tests pin each:
the latent-scale signs ([`docs/notation.md`](docs/notation.md)), the raw
continuous against one-hot ordinal parent encoding and the log-space ordinal
likelihood ([`docs/model.md`](docs/model.md)), and the seeding
([`docs/code-map.md`](docs/code-map.md)). Read [`docs/architecture.md`](docs/architecture.md)
before you change anything in `src/tramdag/`.

## Releasing

The version is the git tag, through hatch-vcs. `cz bump` derives the next tag
from the conventional commits since the last one and `cz changelog` writes
`CHANGELOG.md` from the same commits; a push of the tag runs
`.github/workflows/release.yaml`, which builds with uv, uploads to PyPI through
trusted publishing and creates a sigstore-signed GitHub release.
