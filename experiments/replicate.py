"""Run every experiment variant and check it against the committed ground truth.

The pre-push hook runs this, so a push that moves a pinned number fails
before it leaves the machine; the experiments workflow repeats the same runs
on CI. The variants are those of ``experiments/*.yaml``, the list the
workflow's planner reads too. The runs go in parallel, at most one per core,
each on one thread. The gate checks the commits that leave the machine, so it
refuses a working tree with uncommitted changes.
"""

# %% imports ---------------------------------------------------------------------------
import os
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import yaml

# %% global variables ------------------------------------------------------------------
HERE = Path(__file__).parent
ENV = os.environ | {"OMP_NUM_THREADS": "1", "MKL_NUM_THREADS": "1"}


# %% private functions -----------------------------------------------------------------
def _variants() -> list[tuple[str, str]]:
    """List ``(script, variant)`` for every variant of every experiment YAML."""
    return [
        (path.stem, variant)
        for path in sorted(HERE.glob("*.yaml"))
        for variant in sorted(
            (yaml.safe_load(path.read_text()) or {}).get("variants", ())
        )
    ]


def _replicate(job: tuple[str, str]) -> str | None:
    """Run one variant and its check; give the failure text, or None."""
    script, variant = job
    name = f"{script.replace('_', '-')}-{variant}"
    for args in ([script, variant], ["check", name]):
        done = subprocess.run(
            [sys.executable, "-m", *args],
            cwd=HERE,
            env=ENV,
            capture_output=True,
            text=True,
            check=False,
        )
        if done.returncode:
            return f"{name}: {' '.join(args)} failed\n{done.stdout}{done.stderr}"
    return None


# %% main ------------------------------------------------------------------------------
if __name__ == "__main__":
    if subprocess.run(["git", "diff", "--quiet", "HEAD"], check=False).returncode:
        sys.exit("uncommitted changes: commit or stash them, the gate checks commits")
    jobs = _variants()
    with ThreadPoolExecutor(min(len(jobs), os.cpu_count() or 1)) as pool:
        failures = [f for f in pool.map(_replicate, jobs) if f]
    print("\n\n".join(failures) or f"all {len(jobs)} replications match")
    sys.exit(1 if failures else 0)
