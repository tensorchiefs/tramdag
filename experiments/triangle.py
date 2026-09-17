"""Replicate the paper's continuous-triangle experiments (Sec. 6.1, App. C.3).

The DGP is ``x1 -> x2 -> x3 <- x1`` with every conditional a logistic-latent
flow, ``h(x3|x1,x2) = 0.63 x3 - 0.2 x1 - f(x2)``. Two model families are
fitted, chosen by the variant:

- ``ls`` — the x2 -> x3 edge is a linear shift. Correct only for the linear
  DGP, where the true weight is +0.3; for a nonlinear ``f`` the linear
  weight is the best linear approximation, so no true value is plotted.
- ``cs`` — the edge is a complex shift, which must converge to ``-f(x2)``
  up to an additive constant (paper Fig. 7 right).

Outputs: the coefficient trajectories (Fig. 14/15), the complex-shift
overlay (Fig. 7 right / 17 left / 18 right) and the observational plus
``do(x1)`` distributions (Fig. 16/17).

Usage (from experiments/):

```
uv run python -m triangle atan-cs
```
"""

# %% imports ---------------------------------------------------------------------------
from __future__ import annotations

from functools import partial

from common import (
    cli,
    figure_specs,
    load_variant,
    make_output_dir,
    save_metrics,
    write_report,
)
from helpers import (
    compare_do_x1,
    cs_curve_error,
    fit_paper,
    framework_figures,
    plot_trajectories,
    snapshot,
    true_coefficients,
)
from simulations.triangle import TriangleContinuous


# %% public functions ------------------------------------------------------------------
def run(variant: str) -> dict:
    """Run one variant end to end and give its metrics."""
    config = load_variant(__file__, variant)
    out = make_output_dir(__file__, f"triangle-{variant}")
    figs = figure_specs(config)

    generator = TriangleContinuous(f=config["f"], seed=config["dgp_seed"])
    print(
        f"fitting triangle/{config['f']} with a {config['shift']} shift on "
        f"n={config['n_train']} for {config['fit_kwargs']['epochs']} epochs "
        f"at lr {config['learning_rate']:g} ..."
    )
    train = generator.observational(config["n_train"])
    val = generator.observational(config["n_val"], seed_offset=1)
    flow, trajectory, fit_seconds = fit_paper(
        train,
        val,
        config,
        out,
        record=partial(snapshot, shift=config["shift"]),
    )

    figures = framework_figures(flow, val, out, figs, config["sample_seed"])
    truths = true_coefficients(config)
    plot_trajectories(
        trajectory,
        truths,
        out / "plots" / "coefficients.png",
        figs["coefficients.png"]["title"],
    )
    figures.append("coefficients.png")

    metrics = {key: value for key, value in trajectory[-1].items() if key != "epoch"}
    metrics["val_nll_x3"] = float(flow.nll(val)["x3"])
    metrics["fit_seconds"] = fit_seconds

    if config["shift"] == "cs":
        # which paper figure this is depends on f (7 right for atan, 17 for
        # the misspecified linear case, 18 for sin) — see PAPER_COVERAGE.md
        metrics["cs_curve_max_abs_err"] = cs_curve_error(
            flow,
            generator,
            config,
            out,
            figs["cs_curve.png"]["title"],
        )
        figures.append("cs_curve.png")

    # L1 (observational fit) and L2 (interventional) distributions
    metrics.update(
        compare_do_x1(
            generator,
            flow,
            config,
            out,
            ordinal_levels={},
            title=figs["distributions.png"]["title"],
        )
    )
    figures.append("distributions.png")

    save_metrics(out, metrics)
    write_report(
        out,
        f"triangle: f = {config['f']}, {config['shift']} shift",
        f"Paper Sec. 6.1 and App. C.3: the continuous triangle x1 -> x2 -> x3 <- x1 "
        f"with f = {config['f']} on the x2 -> x3 edge, fitted with a "
        f"{config['shift']} shift. True beta12 = {truths['beta12']:+.1f}, "
        f"beta13 = {truths['beta13']:+.1f}.",
        metrics,
        {name: figs[name] for name in figures},
        # the do(x1) DGP mean is this run's Monte-Carlo truth for the flow's
        truths={**truths, "mean_x3_flow_do_x1": metrics["mean_x3_dgp_do_x1"]},
    )
    print(f"-> {out}")
    return metrics


# %% main ------------------------------------------------------------------------------
if __name__ == "__main__":
    run(cli(__file__, __doc__))
