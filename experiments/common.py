"""The output plumbing every experiment area shares.

Experiments live in one directory per area — ``paper/`` (the replications of
arXiv:2503.16206), ``benchmarks/`` (training and machine speed) and ``misc/``
(everything else, currently the classical-MLE validation). ``paper`` and
``misc`` each own their ``data/``, ``ground_truth/``, ``results/``, ``tests/``
and whatever helpers only they need. ``benchmarks`` measures speed on the other
two's data, so it reads theirs and pins no ground truth of its own.

What is left here is the output layout the experiments workflow reads:
``results/<name>/`` with ``metrics.json``, ``report.md`` and ``plots/``.

Reading a script's YAML file is here. Checking the section it yields is not.
``load_variant`` parses the file and gives the variant's section.
``paper/tests/test_configs.py`` checks that the script reads every key in it.

Every function takes the calling script's ``__file__``, so paths resolve
inside that script's own area with no directory names written in the code.

Run an experiment as a module, from ``experiments/``:

```
uv run python -m paper.triangle atan-cs
uv run python -m check paper triangle-atan-cs
```
"""

# %% imports ---------------------------------------------------------------------------
from __future__ import annotations

import argparse
import json
from pathlib import Path

import yaml


# %% private functions -----------------------------------------------------------------
def _variants(script: str) -> dict:
    """Parse the script's sibling YAML file and give its ``variants`` mapping.

    ``argparse`` takes its choices from the keys, so adding a variant to the
    file is enough to make it runnable. A missing file raises here, naming the
    path it looked for.
    """
    return yaml.safe_load(Path(script).resolve().with_suffix(".yaml").read_text())[
        "variants"
    ]


def _report_row(name: str, value, truths: dict) -> str:
    """One metric row; with truths, a fitted-vs-true row where one exists."""

    def fmt(v):
        return f"{v:+.4f}" if isinstance(v, float) else f"{v}"

    if not truths:
        return f"| `{name}` | {fmt(value)} |"
    if name not in truths:
        return f"| `{name}` | {fmt(value)} |  |  |"
    err = abs(float(value) - float(truths[name]))
    return f"| `{name}` | {fmt(value)} | {fmt(truths[name])} | {err:.4f} |"


# %% public functions ------------------------------------------------------------------
def load_variant(script: str, variant: str) -> dict:
    """Read one variant's hyperparameters from the script's own YAML file.

    The file lists every value the experiment uses, one section per variant
    under ``variants:``. Shared values are written once under an anchor and
    merged with YAML's ``<<`` key, so the merge is visible in the file
    instead of happening in code.

    Parameters
    ----------
    script : str
        The calling script's ``__file__``. The config is the sibling file
        with the same stem and a ``.yaml`` suffix.
    variant : str
        Key under ``variants:``.

    Returns
    -------
    dict
        The variant's hyperparameters.

    Raises
    ------
    FileNotFoundError
        If the config file does not exist, naming the path.
    """
    return dict(_variants(script)[variant])


def cli(script: str, doc: str) -> str:
    """Parse an experiment script's command line and give the chosen variant.

    The one positional argument is the variant; its choices come from the
    script's YAML file and the description from the first line of its
    module docstring.
    """
    parser = argparse.ArgumentParser(description=doc.splitlines()[0])
    parser.add_argument(
        "variant",
        choices=sorted(_variants(script)),
        help="which variant to run; hyperparameters live in the sibling YAML file",
    )
    return parser.parse_args().variant


def make_output_dir(script: str, name: str) -> Path:
    """Create ``<area>/results/<name>/plots/`` and give the results directory."""
    out = Path(script).resolve().parent / "results" / name
    (out / "plots").mkdir(parents=True, exist_ok=True)
    return out


def save_metrics(out: Path, metrics: dict) -> None:
    """Write the numbers the ground-truth check reads."""
    (out / "metrics.json").write_text(json.dumps(metrics, indent=2, default=float))


def write_report(
    out: Path, title: str, metrics: dict, figures: list[str], truths: dict | None = None
) -> None:
    """Write ``report.md``: the metrics table and the figures.

    The experiments workflow posts this file as a commit comment, so the
    figure links are plain relative paths that ``cml comment`` resolves and
    uploads.

    Parameters
    ----------
    out : Path
        The experiment's results directory.
    title : str
        Heading of the report.
    metrics : dict
        Flat ``{name: value}`` mapping, as written to ``metrics.json``.
    figures : list[str]
        File names under ``plots/``, in the order they should appear.
    truths : dict | None, optional
        ``{metric_name: true_value}`` — the DGP/analytic ground truth for the
        metrics that have one. Those rows gain a truth and an ``|err|``
        column, so the commit comment shows fitted-vs-true at a glance.
    """
    truths = truths or {}
    lines = [f"## {title}", ""]
    if truths:
        # no pipes in cell text: cml strips the backslash escapes, which
        # desyncs the header from the |---| separator row
        lines += ["| metric | value | DGP truth | abs. error |", "|---|---|---|---|"]
    else:
        lines += ["| metric | value |", "|---|---|"]
    lines += [_report_row(name, value, truths) for name, value in metrics.items()]
    lines.append("")
    for figure in figures:
        lines += [f"![{figure}](plots/{figure})", ""]
    (out / "report.md").write_text("\n".join(lines))
