"""All-`ls` causal flow (this package's analog of the original `md_dag_ls` run).

Every edge is a linear shift, so each node-conditional is a classical
(proportional-odds / linear transformation) model.

Default data is the public synthetic cohort (`magic-mrclean/nl`); pass a source
to switch, e.g. the private clinical data:

    uv run python all_ls_flow.py                  # synthetic (default)
    uv run python all_ls_flow.py magic-mrclean/ls # synthetic, linear variant
    uv run python all_ls_flow.py magic            # private clinical cohort
"""

import sys

from common import DEFAULT_SOURCE, run_experiment

if __name__ == "__main__":
    source = sys.argv[1] if len(sys.argv) > 1 else DEFAULT_SOURCE
    name = "all_ls" if source == "magic" else f"all_ls_{source.split('/')[-1]}"
    run_experiment(name, style="ls", source=source)
