# CLAUDE.md

`tramdag` is the PyTorch reference implementation of TRAM-DAGs (Sick & Dürr,
CLeaR 2025): one triangular normalizing flow whose sparsity is a causal DAG,
built on zuko. `README.md` says what it is; this file says how to work in it.

## Commands

```bash
uv sync                                        # install from the pinned uv.lock
uv run pytest -q -m "not slow"                 # fast suite (PR CI)
uv run pytest -q                               # full suite (nightly)
uvx ruff check . && pre-commit run --all-files # the 18 hooks CI enforces
uv run mkdocs build --strict                   # docs; README.md becomes index.md
MPLBACKEND=Agg uv run python notebooks/<name>.py   # a notebook, headless
cd experiments && uv run python -m triangle atan-cs   # one replication
cd experiments && uv run python -m check triangle-atan-cs  # vs ground truth
```

## Layout

- `src/tramdag/` is framework code only. `spec.py` declares, `modules.py`
  runs, `nodes.py` holds the two node kinds, `flow.py` composes `FitMixin`
  and `ReadoutsMixin`. `docs/architecture.md` and `docs/code-map.md` are the
  implementation documentation; keep them current.
- `tests/` measures against three inline DGPs in `conftest.py` and never
  imports `experiments/`.
- `experiments/` holds the paper replications and their frozen data; each
  variant's YAML carries the whole model and recipe, the code has no
  defaults. `experiments/data/` is a contract: new seed or equations means a
  new folder.
- `notebooks/*.py` are jupytext sources; only `demo_tram_dag_colab.ipynb` is
  tracked and is regenerated from its `.py`.
- `docs/` are the guides. Each subject has one home (`README.md` lists them);
  examples live in the notebooks only; docstrings document functionality and
  implementation, not background.

## Conventions

- A term is a `Term` subclass with `name` and `module` as class attributes;
  its options are the keyword arguments of `__init__` assigned to `self`, so
  `options()` is `vars(self)`. No registries, no shared base classes between
  terms, no compatibility shims: prefer built-in Python mechanics and explicit
  repeated signatures.
- Node-kind branching lives only in the `Node` methods `encode`, `log_prob`,
  `sample`, `abduct` and `marginal_theta`.
- Latent-scale signs: continuous adds the shift, ordinal subtracts it
  (`docs/notation.md`). The ordinal likelihood stays in log space
  (`docs/model.md`). Parents enter raw or one-hot; seeding happens at
  construction (`docs/code-map.md`).
- Modules keep the `# %%` section markers, the NumPy docstrings and the
  complexipy limits of `CONTRIBUTING.md`.
- Error messages: lowercase, values as `{name!r}`, no trailing period unless
  several sentences.
- Commits follow conventional commits; `cz bump` tags the release and
  `cz changelog` writes `CHANGELOG.md`.

## Hard rules

- Never edit `experiments/*/data/`, test assertion values, or model maths and
  defaults without an explicit request. Numerics stay bit-identical.
- Before adding a mechanism for a rare edge case, discuss it first. The
  simplest solution without special treatment wins.
- Validate a causal feature against known truth, never against "runs".
