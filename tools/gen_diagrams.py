"""Regenerate the auto-generated architecture views in docs/architecture.md.

Two sources, both from the code itself:
- pyreverse (``uvx --from pylint pyreverse -o mmd``) for the package UML and
  the class UML (``-k``: names and inheritance only — full attributes over
  ~30 classes are unreadable);
- a ``sys.setprofile`` trace of one flow construction and one ``fit`` on a
  3-node SI/LS/CS/VC spec for the call graphs (tramdag-internal edges only).

Usage:  uv run python tools/gen_diagrams.py
Rewrites everything between the AUTOGEN markers in docs/architecture.md.
"""

# %% imports ---------------------------------------------------------------------------
import re
import subprocess
import sys
import tempfile
from collections import Counter, defaultdict
from pathlib import Path

# %% global variables ------------------------------------------------------------------
ROOT = Path(__file__).resolve().parents[1]
PKG = "/src/tramdag/"
SKIP = {"<genexpr>", "<listcomp>", "<dictcomp>", "<setcomp>", "<lambda>"}


# %% private functions -----------------------------------------------------------------
def _qualname(code) -> str | None:
    """Name a frame's function as ``module.qualname``, or None if it is foreign.

    Foreign means outside the package, or one of the comprehension frames in
    ``SKIP`` — neither belongs in a call graph of the package's own edges.
    """
    if PKG not in code.co_filename or code.co_name in SKIP:
        return None
    return f"{code.co_filename.split(PKG)[1].removesuffix('.py')}.{code.co_qualname}"


def _is_edge(caller: str | None, callee: str) -> bool:
    """Say whether a traced pair belongs in the call graph.

    Three kinds are dropped: an edge whose caller was foreign code, a
    self-call, and a dunder other than a constructor.
    """
    if caller is None or caller == callee:
        return False
    leaf = callee.rsplit(".", 1)[-1]
    return not leaf.startswith("__") or leaf == "__init__"


def _profiler(edges: Counter, stack: list):
    """Give a ``sys.setprofile`` hook that counts package-internal call edges.

    ``stack`` carries one entry per active frame, so a return always pops what
    a call pushed. A frame outside the package inherits its caller's name,
    which keeps the depth aligned without inventing an edge.
    """

    def prof(frame, event, arg):
        if event == "return":
            if stack:
                stack.pop()
            return
        if event != "call":
            return
        me = _qualname(frame.f_code)
        if me is None:
            stack.append(stack[-1] if stack else None)
            return
        if stack:
            edges[(stack[-1], me)] += 1
        stack.append(me)

    return prof


# %% public functions ------------------------------------------------------------------
def trace(fn) -> Counter:
    """Record tramdag-internal caller->callee edges of one call."""
    edges: Counter = Counter()
    sys.setprofile(_profiler(edges, []))
    try:
        fn()
    finally:
        sys.setprofile(None)
    return Counter({pair: n for pair, n in edges.items() if _is_edge(*pair)})


def flowchart(edges: Counter, drop=()) -> str:
    """Render traced edges as a mermaid flowchart, one subgraph per module."""
    ids: dict = {}
    by_mod = defaultdict(set)
    body = []
    for (a, b), n in sorted(edges.items()):
        if any(a.startswith(p) or b.startswith(p) for p in drop):
            continue
        for name in (a, b):
            ids.setdefault(name, f"n{len(ids)}")
            by_mod[name.split(".")[0]].add(name)
        arrow = f' -- "{n}x" -->' if n > 1 else " -->"
        body.append(f"    {ids[a]}{arrow} {ids[b]}")
    lines = ["flowchart LR"]
    for mod in sorted(by_mod):
        lines.append(f"  subgraph {mod}")
        lines += [
            f'    {ids[nm]}["{nm.split(".", 1)[1]}"]' for nm in sorted(by_mod[mod])
        ]
        lines.append("  end")
    return "\n".join(lines + body)


def main() -> None:
    """Trace, generate, and rewrite the AUTOGEN block."""
    import numpy as np
    import pandas as pd

    from tramdag import CS, LS, SI, VC, CausalFlowDAG, ContinuousNode, OrdinalNode
    from tramdag.callbacks import EarlyStopping

    with tempfile.TemporaryDirectory() as td:
        subprocess.run(
            [
                "uvx",
                "--from",
                "pylint",
                "pyreverse",
                "-o",
                "mmd",
                "-k",  # class names + inheritance only
                "-p",
                "tramdag",
                str(ROOT / "src/tramdag"),
            ],
            cwd=td,
            check=True,
            capture_output=True,
        )
        packages = (Path(td) / "packages_tramdag.mmd").read_text()
        classes = (Path(td) / "classes_tramdag.mmd").read_text()

    rng = np.random.default_rng(0)
    n = 300
    x1 = rng.normal(size=n)
    t = (rng.random(n) < 1 / (1 + np.exp(-x1))).astype(int)
    y = -1.0 * t + 0.5 * x1 + rng.logistic(size=n)
    df = pd.DataFrame({"x1": x1, "t": t, "y": y})
    spec = {
        "x1": ContinuousNode(),
        "t": OrdinalNode(2, [LS("x1")]),
        "y": ContinuousNode(SI() + CS("x1") + VC("x1", t="t", center=False)),
    }
    holder: dict = {}
    construct = trace(lambda: holder.setdefault("flow", CausalFlowDAG(spec, seed=0)))
    fit = trace(
        lambda: holder["flow"].fit(
            df,
            epochs=3,
            batch_size=150,
            validation_split=0.2,
            callbacks=EarlyStopping(),
        )
    )

    marker = "<!-- AUTOGEN:diagrams (tools/gen_diagrams.py) — do not edit by hand -->"
    section = f"""{marker}
## Generated views

To regenerate this section, run ``uv run python tools/gen_diagrams.py``.

The package UML and the class UML come from pyreverse. The class diagram
carries names and inheritance only. The call graphs come from a profile trace
of one flow construction and one three-epoch ``fit`` on a 3-node SI/LS/CS/VC
spec. The graphs keep tramdag-internal edges only, and ``3x`` means once per
node.

### Package UML (pyreverse)

```mermaid
{packages.strip()}
```

### Class UML (pyreverse)

The built-in terms are conditioners with a term-hook mixin: each concrete
term class inherits its network from ``conditioners`` and its contract
from ``ShiftTerm``/``InterceptTerm``.

```mermaid
{classes.strip()}
```

### Call graph — flow construction (traced)

```mermaid
{flowchart(construct, drop=("spec.Term.__getattr__", "spec._option_defaults"))}
```

### Call graph — one fit (traced)

```mermaid
{flowchart(fit)}
```
<!-- AUTOGEN:end -->"""

    doc = ROOT / "docs/architecture.md"
    s = doc.read_text()
    if "<!-- AUTOGEN:diagrams" in s:
        s = re.sub(r"<!-- AUTOGEN:diagrams[\s\S]*?<!-- AUTOGEN:end -->", section, s)
    else:
        s = s.rstrip() + "\n\n" + section + "\n"
    doc.write_text(s)
    print(
        f"architecture.md updated: construct {len(construct)} edges, "
        f"fit {len(fit)} edges"
    )


# %% main ------------------------------------------------------------------------------
if __name__ == "__main__":
    main()
