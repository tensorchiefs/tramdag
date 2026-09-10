"""Seeded state_dict-diff smoke: the RNG-stream and key-layout tripwire.

Builds one flow per inline DGP spec at a fixed seed and compares every parameter
bit-for-bit against a recorded baseline. Any change that reorders parameter
construction fails here at the commit that caused it. That is the silent
killer: the tests pass and the pinned experiment centers move.

Usage:  uv run python tests/tools/statedict_smoke.py record   # write baseline
        uv run python tests/tools/statedict_smoke.py check    # compare
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
        "x2": ContinuousNode([LS("x1")]),
        "t": OrdinalNode(2, [LS("x1"), LS("x2")]),
        "y": OrdinalNode(4, [LS("x1"), LS("x2"), LS("t")]),
    },
    "flexible": lambda: {
        "x1": ContinuousNode(),
        "x2": ContinuousNode([CI("x1")]),
        "x3": ContinuousNode([SI(transform="spline"), CS("x1", "x2")]),
    },
    "vc": lambda: {
        "X1": ContinuousNode(),
        "T": OrdinalNode(2, [LS("X1")]),
        "Y": ContinuousNode([CS("X1"), VC("X1", t="T")]),
    },
    "additive_ci": lambda: {
        "a": ContinuousNode(),
        "b": ContinuousNode(),
        "y": ContinuousNode([CI("a", "b", allow_interaction=False)]),
    },
}


# %% private functions -----------------------------------------------------------------
def _diff(name: str, params: dict, cur: dict | None) -> list[str]:
    if cur is None:
        return [f"{name}: spec no longer builds"]
    if set(cur) != set(params):
        gone = sorted(set(params) - set(cur))
        new = sorted(set(cur) - set(params))
        return [f"{name}: key layout changed (-{gone} +{new})"]
    return [
        f"{name}: {k} differs" for k, v in params.items() if not torch.equal(v, cur[k])
    ]


# %% public functions ------------------------------------------------------------------
def build() -> dict:
    out = {}
    for name, spec in SPECS.items():
        flow = CausalFlowDAG(spec(), seed=0)
        out[name] = {k: v.clone() for k, v in flow.state_dict().items()}
    return out


def main() -> int:
    mode = sys.argv[1] if len(sys.argv) > 1 else "check"
    current = build()
    if mode == "record":
        torch.save(current, BASELINE)
        print(f"baseline recorded: {BASELINE}")
        return 0
    baseline = torch.load(BASELINE, weights_only=True)
    bad = [
        b
        for name, params in baseline.items()
        for b in _diff(name, params, current.get(name))
    ]
    if bad:
        print("STATE-DICT SMOKE FAILED — RNG stream or key layout moved:")
        print("\n".join(f"  {b}" for b in bad))
        return 1
    print("state-dict smoke OK: all parameters bit-identical to baseline")
    return 0


# %% main ------------------------------------------------------------------------------
if __name__ == "__main__":
    sys.exit(main())
