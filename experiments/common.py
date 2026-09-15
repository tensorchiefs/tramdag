"""The output plumbing the experiment scripts share.

The experiments workflow reads ``results/<name>/`` with ``metrics.json``,
``report.md`` and ``plots/``, which the functions here write.

Reading a script's YAML file is here. Checking the section it yields is not.
``load_variant`` parses the file and gives the variant's section.
``tests/test_configs.py`` checks that the script reads every key in it.

Every function takes the calling script's ``__file__``, so paths resolve next
to that script with no directory names written in the code.

Run an experiment as a module, from ``experiments/``:

```
uv run python -m triangle atan-cs
uv run python -m check triangle-atan-cs
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


def figure_specs(config: dict, **extra) -> dict[str, dict[str, str]]:
    """Give the variant's ``figures`` with titles and captions filled in.

    Every value is formatted with the variant's own keys, so a title in the
    YAML may say ``{f}`` or ``{shift}``; ``extra`` adds run-time values such
    as an observation the title quotes.
    """
    values = {**config, **extra}
    return {
        name: {field: text.format(**values) for field, text in spec.items()}
        for name, spec in config["figures"].items()
    }


def make_output_dir(script: str, name: str) -> Path:
    """Create ``results/<name>/plots/`` and give the results directory."""
    out = Path(script).resolve().parent / "results" / name
    (out / "plots").mkdir(parents=True, exist_ok=True)
    return out


def save_metrics(out: Path, metrics: dict) -> None:
    """Write the numbers the ground-truth check reads."""
    (out / "metrics.json").write_text(json.dumps(metrics, indent=2, default=float))


def write_report(
    out: Path,
    title: str,
    summary: str,
    metrics: dict,
    figures: dict[str, dict[str, str]],
    truths: dict | None = None,
) -> None:
    """Write ``report.md``: what the run is, its metrics table, its figures.

    The experiments workflow posts this file as a commit comment, so the
    figure links are plain relative paths that ``cml comment`` resolves and
    uploads.

    Parameters
    ----------
    out : Path
        The experiment's results directory.
    title : str
        Heading of the report.
    summary : str
        One or two sentences on what the run replicates.
    metrics : dict
        Flat ``{name: value}`` mapping, as written to ``metrics.json``.
    figures : dict[str, dict[str, str]]
        ``{file name under plots/: {"title": ..., "caption": ...}}`` in the
        order they should appear; [`figure_specs`][] builds it from the YAML.
    truths : dict | None, optional
        ``{metric_name: true_value}`` — the DGP/analytic ground truth for the
        metrics that have one. Those rows gain a truth and an ``|err|``
        column, so the commit comment shows fitted-vs-true at a glance.
    """
    truths = truths or {}
    about = (
        "The numbers `metrics.json` holds and `check.py` compares against the "
        "committed ground truth. Where the DGP truth is known, the row also "
        "shows it and the absolute error."
    )
    lines = [f"## {title}", "", summary, "", "### Metrics", "", about, ""]
    if truths:
        # no pipes in cell text: cml strips the backslash escapes, which
        # desyncs the header from the |---| separator row
        lines += ["| metric | value | DGP truth | abs. error |", "|---|---|---|---|"]
    else:
        lines += ["| metric | value |", "|---|---|"]
    lines += [_report_row(name, value, truths) for name, value in metrics.items()]
    lines += ["", "### Figures", ""]
    for figure, spec in figures.items():
        lines += [
            f"#### {spec['title']}",
            "",
            spec["caption"],
            "",
            f"![{figure}](plots/{figure})",
            "",
        ]
    (out / "report.md").write_text("\n".join(lines))
