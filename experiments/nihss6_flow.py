"""Flexible causal flow — the configuration featured in the paper (`nihss6`).

Per-edge terms as in nihss6/configuration.json: Age enters every child via a
complex intercept ('ci'), mRS_pre via linear shifts ('ls'), NIHSSa via complex
shifts ('cs'), T -> mRS_3m via a linear shift ('ls').

Default data is the public synthetic cohort (`magic-mrclean/nl`); pass a source
to switch, e.g. `magic` for the private clinical cohort:

    uv run python nihss6_flow.py                  # synthetic (default)
    uv run python nihss6_flow.py magic            # private clinical cohort
"""

import sys

from common import DEFAULT_SOURCE, run_experiment

if __name__ == "__main__":
    source = sys.argv[1] if len(sys.argv) > 1 else DEFAULT_SOURCE
    name = "nihss6" if source == "magic" else f"nihss6_{source.split('/')[-1]}"
    run_experiment(name, style="flexible", source=source)
