"""Observational ITE benchmark DGP (Krause, MA experiment 4).

A 7-variable mediation SCM with a binary treatment, used to study
individual-treatment-effect (ITE) estimation with an S-learner TRAM-DAG in an
*observational* setting (treatment confounded by X1, X2). Ported from
``code/ITE_observational_simulation.R`` of github.com/mikekr97/MA_Mike.

DAG (topological order ``X1 X2 X3 Tr X5 X6 Y``)::

    X1, X2, X3 ~ N(0, Sigma)            compound-symmetric, corr rho=0.1
    Tr ~ Bernoulli(sigmoid(0.5 - 0.5 X1 + 0.3 X2))        binary treatment
    X5 :  2.5 X5 = logit(U5) - 0.8 Tr                     Colr mediator
    X6 :  4.0 X6 = logit(U6) + 0.5 X5                     Colr mediator
    Y  :  h_y(Y) = logit(U7) - [ beta_t Tr + X.beta_X
                                 + (-0.9 X2 + 0.7 X3) Tr ]

with ``X = (X1, X2, X3, X5, X6)``, ``beta_X = (-0.5, 0.5, 0.2, -0.6, 0.4)`` and a
**nonlinear baseline** transform ``h_y(y) = tan(y/2)/0.2`` on ``[-2, 2]`` with
slope-matched linear extrapolation outside (so ``h_y_inverse(z) = 2 atan(0.2 z)``
in the core). All latents are logistic (``logit(U)``) — i.e. the DGP lives
*inside* the flow's standard-logistic family; the only thing to learn is the
shapes/interactions.

Four scenarios toggle the outcome's treatment terms (``scenario`` 1..4):

==========  ===========  ================  =========================
scenario    beta_t       beta_TX (X2,X3)   meaning
==========  ===========  ================  =========================
1           1.5          (-0.9, 0.7)       main effect + interaction
2           1.5          (0, 0)            main effect only
3           0            (-0.9, 0.7)       interaction only
4           0            (0, 0)            null (no treatment effect)
==========  ===========  ================  =========================

Because the treatment also drives the mediators X5 -> X6 -> Y, an individual's
effect flows through both the direct path and the mediation path. The SCM exposes
two notions of per-individual ground truth (shared exogenous noise across the two
treatment arms):

- ``ITE_true``   = Y(Tr=1) - Y(Tr=0)   at the *observed* latent U7
- ``ITE_median`` = the same contrast at the *median* latent (logit = 0)

and ``ATE`` = population mean of either. The featured/frozen dataset is
**scenario 1** (heterogeneous ITEs).

CLI::

    uv run python -m tramdag.simulations.ite_observational \\
        --out data/ite-observational --seed 123 --scenario 1
"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

COLUMNS = ("X1", "X2", "X3", "Tr", "X5", "X6", "Y")
_SCENARIOS = {1: (1.5, (-0.9, 0.7)), 2: (1.5, (0.0, 0.0)),
              3: (0.0, (-0.9, 0.7)), 4: (0.0, (0.0, 0.0))}
_BETA_X = np.array([-0.5, 0.5, 0.2, -0.6, 0.4])   # X1, X2, X3, X5, X6
_DIVISOR = 0.2                                     # h_y core: tan(y/2)/0.2
_BOUND = 2.0                                       # h_y core domain [-2, 2]


def _logit(u: np.ndarray) -> np.ndarray:
    return np.log(u) - np.log1p(-u)


def _sigmoid(x: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-x))


# baseline transform h_y and its inverse (matching ITE_utils.R) ----------------
_Y_AT_BOUND = np.tan(_BOUND / 2) / _DIVISOR                       # h_y(2)
_SLOPE_AT_BOUND = (1.0 / _DIVISOR) * 0.5 / np.cos(_BOUND / 2) ** 2  # h_y'(2)


def h_y(x: np.ndarray) -> np.ndarray:
    """Nonlinear baseline transform; ``tan(x/2)/0.2`` on [-2, 2], linear outside."""
    x = np.asarray(x, dtype=float)
    core = np.tan(np.clip(x, -_BOUND, _BOUND) / 2) / _DIVISOR
    left = -_Y_AT_BOUND + _SLOPE_AT_BOUND * (x + _BOUND)
    right = _Y_AT_BOUND + _SLOPE_AT_BOUND * (x - _BOUND)
    return np.where(x < -_BOUND, left, np.where(x > _BOUND, right, core))


def h_y_inverse(y: np.ndarray) -> np.ndarray:
    """Inverse of :func:`h_y`; ``2 atan(0.2 y)`` in the core, linear in the tails."""
    y = np.asarray(y, dtype=float)
    core = 2.0 * np.arctan(_DIVISOR * np.clip(y, -_Y_AT_BOUND, _Y_AT_BOUND))
    left = (y + _Y_AT_BOUND) / _SLOPE_AT_BOUND - _BOUND
    right = (y - _Y_AT_BOUND) / _SLOPE_AT_BOUND + _BOUND
    return np.where(y <= -_Y_AT_BOUND, left, np.where(y >= _Y_AT_BOUND, right, core))


@dataclass
class ITEObservational:
    """SCM generator for the observational ITE benchmark (default scenario 1)."""

    seed: int = 123
    rho: float = 0.1
    scenario: int = 1

    @property
    def beta_t(self) -> float:
        return _SCENARIOS[self.scenario][0]

    @property
    def beta_TX(self) -> np.ndarray:
        return np.array(_SCENARIOS[self.scenario][1])

    def draw_latents(self, n: int, rng: np.random.Generator) -> dict[str, np.ndarray]:
        sigma = np.full((3, 3), self.rho)
        np.fill_diagonal(sigma, 1.0)
        return {
            "x123": rng.multivariate_normal(np.zeros(3), sigma, size=n),
            "u_tr": rng.uniform(size=n),
            "u5": rng.uniform(size=n),
            "u6": rng.uniform(size=n),
            "u7": rng.uniform(size=n),
        }

    def _logit_outcome(self, X: np.ndarray, Tr: np.ndarray) -> np.ndarray:
        """The outcome shift  beta_t Tr + X.beta_X + (beta_TX . (X2,X3)) Tr."""
        inter = (X[:, 1] * self.beta_TX[0] + X[:, 2] * self.beta_TX[1]) * Tr
        return self.beta_t * Tr + X @ _BETA_X + inter

    def simulate(self, n: int | None = None, *, rng: np.random.Generator | None = None,
                 do: dict[str, float] | None = None,
                 latents: dict[str, np.ndarray] | None = None) -> pd.DataFrame:
        """Sample the SCM. ``do`` clamps a variable and mutilates its parents;
        downstream variables are recomputed from the shared ``latents`` (so two
        ``simulate`` calls differing only in ``do={"Tr": .}`` are counterfactual
        siblings of each other)."""
        do = do or {}
        if latents is None:
            if n is None:
                raise ValueError("provide either n or latents")
            rng = rng or np.random.default_rng(self.seed)
            latents = self.draw_latents(n, rng)
        n = len(latents["u7"])

        def clamp_or(name, value):
            return np.full(n, float(do[name])) if name in do else value

        x123 = latents["x123"]
        X1 = clamp_or("X1", x123[:, 0])
        X2 = clamp_or("X2", x123[:, 1])
        X3 = clamp_or("X3", x123[:, 2])

        prob_tr = _sigmoid(0.5 - 0.5 * X1 + 0.3 * X2)
        Tr = clamp_or("Tr", (latents["u_tr"] < prob_tr).astype(float))

        X5 = clamp_or("X5", (_logit(latents["u5"]) - 0.8 * Tr) / 2.5)
        X6 = clamp_or("X6", (_logit(latents["u6"]) + 0.5 * X5) / 4.0)

        X = np.column_stack([X1, X2, X3, X5, X6])
        Y = clamp_or("Y", h_y_inverse(_logit(latents["u7"]) - self._logit_outcome(X, Tr)))
        return pd.DataFrame({"X1": X1, "X2": X2, "X3": X3, "Tr": Tr,
                             "X5": X5, "X6": X6, "Y": Y})

    # ----------------------------------------------------------------- datasets
    def observational(self, n: int, seed_offset: int = 0) -> pd.DataFrame:
        rng = np.random.default_rng(self.seed + 1 + seed_offset)
        return self.simulate(n, rng=rng)

    def with_truth(self, n: int, seed_offset: int = 0) -> pd.DataFrame:
        """Observational rows plus aligned per-individual ground truth columns
        ``ITE_true`` and ``ITE_median`` (shared latents across the two arms)."""
        rng = np.random.default_rng(self.seed + 1 + seed_offset)
        latents = self.draw_latents(n, rng)
        obs = self.simulate(latents=latents)
        cf1 = self.simulate(latents=latents, do={"Tr": 1.0})
        cf0 = self.simulate(latents=latents, do={"Tr": 0.0})
        out = obs.copy()
        out["ITE_true"] = (cf1["Y"] - cf0["Y"]).to_numpy()
        out["ITE_median"] = self._ite_median(latents)
        return out

    def _ite_median(self, latents: dict[str, np.ndarray]) -> np.ndarray:
        """ITE at the median latent (logit U7 = 0), with the mediators taken under
        each treatment arm — matches the R ``ITE_median``."""
        n = len(latents["u7"])
        x123 = latents["x123"]
        X1, X2, X3 = x123[:, 0], x123[:, 1], x123[:, 2]
        out = []
        for tr in (1.0, 0.0):
            X5 = (_logit(latents["u5"]) - 0.8 * tr) / 2.5
            X6 = (_logit(latents["u6"]) + 0.5 * X5) / 4.0
            X = np.column_stack([X1, X2, X3, X5, X6])
            out.append(h_y_inverse(0.0 - self._logit_outcome(X, np.full(n, tr))))
        return out[0] - out[1]

    # -------------------------------------------------------------- ground truth
    def true_ate(self, mc_n: int = 1_000_000) -> dict:
        """Monte-Carlo ATE from both ITE notions (large sample, fixed seed)."""
        df = self.with_truth(mc_n, seed_offset=777)
        return {"mc_n": mc_n,
                "ate_true": float(df["ITE_true"].mean()),
                "ate_median": float(df["ITE_median"].mean()),
                "ite_true_std": float(df["ITE_true"].std()),
                "ite_median_std": float(df["ITE_median"].std())}


# --------------------------------------------------------------------------- CLI
def main(argv: list[str] | None = None) -> None:
    p = argparse.ArgumentParser(description="Generate the observational ITE data.")
    p.add_argument("--out", type=Path, default=Path("data/ite-observational"))
    p.add_argument("--seed", type=int, default=123)
    p.add_argument("--scenario", type=int, default=1, choices=(1, 2, 3, 4))
    p.add_argument("--n-obs", type=int, default=5000)
    p.add_argument("--mc-n", type=int, default=1_000_000)
    args = p.parse_args(argv)

    gen = ITEObservational(seed=args.seed, scenario=args.scenario)
    args.out.mkdir(parents=True, exist_ok=True)
    obs = gen.observational(args.n_obs)
    obs.to_csv(args.out / "obs.csv", index=False)

    bt, btx = _SCENARIOS[args.scenario]
    truth = {"source": "github.com/mikekr97/MA_Mike code/ITE_observational_simulation.R",
             "seed": args.seed, "scenario": args.scenario, "n_obs": args.n_obs,
             "rho": gen.rho, "beta_t": bt, "beta_TX": list(btx),
             "beta_X": _BETA_X.tolist(),
             "scm": {"Tr": "Bernoulli(sigmoid(0.5 - 0.5 X1 + 0.3 X2))",
                     "X5": "2.5 X5 = logit(U5) - 0.8 Tr",
                     "X6": "4 X6 = logit(U6) + 0.5 X5",
                     "Y": "h_y(Y) = logit(U7) - [beta_t Tr + X.beta_X + (beta_TX.(X2,X3)) Tr]",
                     "h_y": "tan(y/2)/0.2 on [-2,2], linear outside"},
             **gen.true_ate(args.mc_n)}
    (args.out / "truth.json").write_text(json.dumps(truth, indent=2) + "\n")
    print(f"[ite-observational] scenario {args.scenario}  n={len(obs)}  "
          f"Tr rate={obs['Tr'].mean():.3f}  "
          f"ATE_true={truth['ate_true']:+.3f}  ATE_median={truth['ate_median']:+.3f}")


if __name__ == "__main__":
    main()
