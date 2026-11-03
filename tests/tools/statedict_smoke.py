"""Seeded state_dict-diff smoke: the RNG-stream and key-layout tripwire.

Builds one flow per inline DGP spec at a fixed seed and records every parameter
as the baseline. ``tests/test_statedict_stability.py`` compares against it
bit-for-bit, so a change that reorders parameter construction fails at the
commit that caused it. That is the silent killer: the tests pass and the
pinned experiment centers move.

Usage:  uv run python tests/tools/statedict_smoke.py   # write baseline
"""

# %% imports ---------------------------------------------------------------------------
import sys
from pathlib import Path

import torch

from tramdag import CI, CS, LS, SI, VC, CausalFlowDAG, ContinuousNode, OrdinalNode

# %% global variables ------------------------------------------------------------------
BASELINE = Path(__file__).with_name("statedict_baseline.pt")

SPECS = {
    "ls_chain": lambda: {
        "x1": ContinuousNode(),
        "x2": ContinuousNode(LS("x1")),
        "t": OrdinalNode(2, LS("x1") + LS("x2")),
        "y": OrdinalNode(4, LS("x1") + LS("x2") + LS("t")),
    },
    "flexible": lambda: {
        "x1": ContinuousNode(),
        "x2": ContinuousNode(CI("x1")),
        "x3": ContinuousNode(SI(transform="spline") + CS("x1", "x2")),
    },
    "vc": lambda: {
        "X1": ContinuousNode(),
        "T": OrdinalNode(2, LS("X1")),
        "Y": ContinuousNode(CS("X1") + VC("X1", t="T")),
    },
    "additive_ci": lambda: {
        "a": ContinuousNode(),
        "b": ContinuousNode(),
        "y": ContinuousNode(CI("a", "b", allow_interaction=False)),
    },
}


# %% public functions ------------------------------------------------------------------
def build() -> dict:
    out = {}
    for name, spec in SPECS.items():
        flow = CausalFlowDAG(spec(), seed=0)
        out[name] = {k: v.clone() for k, v in flow.state_dict().items()}
    return out


def main() -> int:
    torch.save(build(), BASELINE)
    print(f"baseline recorded: {BASELINE}")
    return 0


# %% main ------------------------------------------------------------------------------
if __name__ == "__main__":
    sys.exit(main())
