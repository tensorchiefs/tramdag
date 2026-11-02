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
`tests/` and `experiments/tests/`.

[`tests/README.md`](tests/README.md) documents what the suite guarantees, what
`slow` marks, how CI splits the runs, and where each reference number comes
from. The policy:

- Validate a new causal feature against the known truth of an inline DGP in
  `tests/conftest.py`, never against "runs without error". If no DGP fits,
  add one.
- Mark a long fit `@pytest.mark.slow` so PR CI stays fast, but never the fit
  that *is* a feature's acceptance measurement.
- A framework test never imports `experiments/`. The research generators and
  their frozen CSVs live there, and the experiments workflow checks them.
- `experiments/data/` is a contract. A new seed or changed equations
  means a new folder, never an edit in place.

## Linting

```bash
uvx ruff check .                    # report
uvx ruff format --diff src tests experiments tools notebooks   # formatting
```

`ruff format` also reformats fenced Python inside markdown, which the
`ruff-format` hook deliberately does not do. The guides wrap code for
reading, not for the formatter, so point the command at the source trees.

Rules live in `pyproject.toml`: ruff's default set plus the extras listed
under `extend-select`, at 88 columns with the numpy docstring convention.
The complexipy hooks gate cognitive complexity. The limit is 15 for `src/`
and `tools/`. The limit is 10 for `experiments/`, `notebooks/` and `tests/`.
Function docstrings are not required in `tests/`, `experiments/` or
`notebooks/`.

`.github/workflows/pre-commit.yaml` enforces the hooks in
`.pre-commit-config.yaml` on every push and pull request. Install
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
`.ipynb`. Git tracks one `.ipynb` on purpose. It is
`demo_tram_dag_colab.ipynb`, because the README's Colab badge links to it.
Regenerate it from the `.py`. See
[`notebooks/README.md`](notebooks/README.md).

## Conventions worth knowing

Four implementation conventions are easy to get wrong, and tests pin each:

- the latent-scale signs ([`docs/notation.md`](docs/notation.md));
- the raw continuous against one-hot ordinal parent encoding
  ([`docs/model.md`](docs/model.md));
- the log-space ordinal likelihood ([`docs/model.md`](docs/model.md));
- the seeding ([`docs/code-map.md`](docs/code-map.md)).

Read [`docs/architecture.md`](docs/architecture.md) before you change
anything in `src/tramdag/`.

## Citations

Every cited work has one entry in
[`docs/references.bib`](docs/references.bib), with its DOI or arXiv link.
The entry takes its metadata from a primary source, such as Crossref or the
proceedings page. Keep the file sorted by first author. Name the key
`<firstauthor><year><word>`, for example `pearl2009causality`. The
References page lists every entry by itself.

- The guides cite with `[@key]` in brackets. Add a locator after a comma,
  as in `[@key, App. E.1]`.
- Separate several keys with semicolons, in alphabetical order, as in
  `[@a; @b]`. The site numbers its footnote links in source order and sorts
  the text, so any other order mismatches the two.
- Do not use a bare `@key`, because it renders the whole entry inline.
- The site and the PDF both format with [`docs/apa.csl`](docs/apa.csl), an
  author-year style. `uv run mkdocs build --strict` fails on an unknown key.
- Code, notebooks and tests cite only where they implement the cited work.
  They cite as plain text in the same form, for example "Robinson (1988)" or
  "(Dandl et al., 2024)". Configurations carry no citations.
- Background belongs in the guides, not in docstrings.

## Tooling and CI

Config files carry no comments. The hooks `no-config-comments` and
`no-experiment-yaml-comments` enforce that. This section holds the reasons.

`pyproject.toml`:

- `dev` is the default `uv sync` group. It holds everything a working checkout
  needs. It pins `ruff` to the rev of the ruff-pre-commit hook. Otherwise `ruff`
  is only a transitive dependency of `python-lsp-ruff`, and `uv run ruff` and
  the hook can disagree about formatting.
- `dev` also holds `shellcheck-py`. actionlint lints the shell of a workflow
  only when shellcheck is on `PATH`. Without it, a local hook run checks less
  than CI does.
- The dependency groups serve one purpose each. `docs` builds the site and the
  PDF. `experiments` runs the replications. `notebooks` runs the jupytext
  sources. `test` runs the suite, with `pytest-cov` for the coverage gate.
- `testpaths` covers `experiments/` too. The `test` group must therefore import
  what those tests import: the generators (scipy), matplotlib through the
  experiment modules, and pyyaml for `experiments/tests/test_configs.py`.
- `pythonpath = ["experiments"]` makes `simulations` and `common` import as they
  do under `python -m triangle` from `experiments/`.
- The experiment tests are cheap numpy-only generator checks. The training runs
  live in the experiments workflow.
- CI passes `-n auto`. It is not in `addopts`, so a bare `pytest` stays serial.
  With 16 workers the full suite takes about 12 minutes against about 24 minutes
  serial on a 32-core machine.
- The coverage gate is 95. The fast suite alone measures 97. The margin absorbs
  run-to-run variation and still fails on an untested new code path.
- The sdist excludes `experiments/` (about 2 MB of frozen CSVs) and `tests/`.
  hatch-vcs uses `no-local-version`, so the version suits PyPI.
- `cz bump` derives the next version from the conventional commits since the
  last tag and creates the tag that the release workflow publishes.
- ruff keeps its default set, which in 0.16 includes S110, BLE001 and TRY004.
  `extend-select` adds to it. The rules from `A` on are clean and bind new code
  only.
- ruff ignores `RUF001` to `RUF003`, because em dashes and Greek letters in
  prose are deliberate (`docs/notation.md`). It ignores `C408`, because the
  experiments' plotting code uses `dict(arrowstyle=...)` as a matplotlib idiom.
  It ignores `SIM108`, because a ternary is not always clearer.
- Notebook prose lives in `# %% [markdown]` cells, where a line is a paragraph,
  so `E501` and the numpydoc summary rules do not apply. The glob
  `**/notebooks/*` matches at any depth, because `docs/notebooks/` holds tracked
  symlinks that ruff follows.
- Scripts and tests document themselves in prose and in `tests/README.md`, so
  `experiments/*` and `tests/*` ignore the docstring rules that they raise.
  Reformatting `tests/` for them is churn.

`.pre-commit-config.yaml`:

- `check-yaml` runs with `--unsafe` and checks syntax only, because `mkdocs.yml`
  needs the `!!python/name` tag of the mermaid fence.
- `pretty-format-toml` excludes `uv.lock`. uv owns that file, and formatting
  reorders thousands of lines.
- codespell ignores three words. `nd` is the `_Node` variable in `flow.py`.
  `continous` is a quote of the filename of the upstream R script in
  `tensorchiefs/tram-dag`, with its typo. `lik` is a quote of the R output `'log
  Lik.'` in a notebook.
- A commit-msg hook receives only the message file. The second codespell hook
  therefore checks the message, and the file excludes do not apply to it.
- complexipy has two tiers. Framework code may be intricate where the maths is.
  Examples and tests must stay simple enough to read as prose. `tools/` is
  developer code, so it belongs to the framework tier. The examples tier skips
  `.ipynb`, which the `.py` sources generate.
- The `commitizen-branch` pre-push hook is not wired up. pre-commit exports
  `PRE_COMMIT_FROM_REF` and `PRE_COMMIT_TO_REF` only when both refs exist. The
  first push of a new branch would hand commitizen the literal
  `$PRE_COMMIT_FROM_REF..$PRE_COMMIT_TO_REF` and abort. The commit-msg hook
  already checks every message.

Workflows:

- `ci.yml` does not pin Linux to the CPU-only torch index in `uv.lock`, because
  GPU machines share that lock file. The default Linux wheel bundles CUDA and is
  large, so the uv cache, keyed on `uv.lock`, keeps runs fast.
- The test matrix runs the oldest and the newest supported Python. CI calls
  `pytest` with no path, so `testpaths` runs `tests/` and the experiment checks.
  `--cov` takes `source` and `fail_under` from `[tool.coverage]`.
- The CI jobs set a timeout. A torch hang would otherwise burn the 6-hour
  default per matrix job. The nightly slow suite needs the headroom.
- The test job syncs with `--no-default-groups`. `dev` is a default group and
  pulls the editor and docs tooling too (144 packages against 45 for `test`
  alone). `uv run` then needs the same flag, or it re-syncs `dev`.
- Torch opens as many intra-op threads as the machine reports, and `-n auto`
  already runs one worker per vCPU. Uncapped, one worker ran 41 minutes CPU in 9
  minutes wall without finishing. With the cap, the whole not-slow suite runs in
  under a minute with `-n auto`.
- The diagrams job regenerates the views in `docs/architecture.md` and fails on
  a diff. A hand-run command produced them, so nothing caught drift: the
  committed copy named `_check_levels` after that method became
  `_check_level_values`. Core dependencies suffice, because the script imports
  tramdag, numpy and pandas and fetches pyreverse through `uvx`.
- `docs.yaml` publishes the site per ref with mike: `main` at the site root as
  `latest`, every `dev-*` branch under its own version. It also builds one PDF
  of the README, the guides and the executed notebooks with pandoc and XeLaTeX.
  The PDF is a convenience. Its step is `continue-on-error`, and the workflow
  republishes the previous copy when it fails. A warning marks that case, so a
  permanently broken PDF build cannot hide behind a green job.
- The PDF staging rewrites every input twice. It drops badges, because a remote
  `.svg` badge makes pandoc emit `\includesvg`, which needs inkscape. It also
  drops the `[x][ref]` cross-references of mkdocstrings. Pandoc prints them raw,
  and the long token overflows its table column.
- `architecture.md` stays out of the PDF. It is mostly mermaid fences, which
  pandoc prints as raw code.
- pandoc uses its `markdown` reader, not `gfm`, because only it derives table
  column widths from the separator row. A wide table then wraps inside `p{}`
  columns. citeproc resolves the `[@key]` citations with the site's `.bib` and
  CSL. The pandoc is the one that `pypandoc-binary` bundles, which the site
  formats with.
- The republish step reads the PDF from `gh-pages` at the ref, not the `latest`
  alias, because that alias is a symlink that `git show` cannot traverse.
- `docs.yaml` has `contents: write` and a full checkout because mike reads and
  writes the `gh-pages` branch. The mike `latest` alias is a symlink, so the
  republish step uses the ref directory.
- `experiments.yaml` runs every experiment as one matrix job and posts the
  report as a commit comment. It trains real models on the protocol of
  `docs/paper-replication.md`, which lists the CI deviations. The matrix comes
  from the YAML configs, so a new variant joins automatically. A
  `workflow_dispatch` may narrow it to one script and variant. Every push runs
  it. Concurrency cancels superseded runs, one run per ref. `fail-fast` is off,
  so one drifting experiment does not hide the rest. The CML action uploads the
  figures to its asset server, so no branch holds generated images.
- The experiments jobs set `UV_NO_DEFAULT_GROUPS` because `dev` drags in editor
  tooling, so the job names the group. They set `OMP_NUM_THREADS` to 2 because
  the runners have 2 cores. The 60-minute timeout fits a slowest job of 11
  minutes, and a hang is a hang.
- The report step runs after a drifted run too, because the figures matter most
  then. It needs the run to have written its report. Earlier failures show as
  the job's own error.
- The output directory of a run is `<script-with-dashes>-<variant>`.
  `make_output_dir` dashes the script name and keeps the variant verbatim.
- `release.yaml` adapts `MArpogaus/minimal_python_project_skeleton`, which uses
  setuptools_scm where this repository uses hatch-vcs. It has no TestPyPI job.
  Tags publish straight to PyPI through trusted publishing, with no token secret
  (`id-token: write`). The checkout fetches full history so hatch-vcs can read
  the tag.

## Releasing

The version is the git tag, through hatch-vcs. `cz bump` derives the next tag
from the conventional commits since the last one. `cz changelog` writes
`CHANGELOG.md` from the same commits. A push of the tag runs
`.github/workflows/release.yaml`. It builds with uv, uploads to PyPI through
trusted publishing and creates a sigstore-signed GitHub release.
