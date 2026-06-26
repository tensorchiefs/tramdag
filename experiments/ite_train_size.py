"""ITE recovery vs training-set size for an all-CI TRAM-DAG.

Companion experiment to ``notebooks/ite_observational.py``: fit the same all-CI
S-learner TRAM-DAG on increasingly large observational samples
(n = 500 .. 20000), evaluate every model on the **same fixed 5000-row test set**,
and report

  * ATE recovery (predicted vs the known true ATE),
  * ITE recovery (correlation and MAE vs the per-individual true ITE), and
  * the **train vs test NLL gap** — the overfitting diagnostic: a flexible
    all-CI model trained on little data fits its train set better than the test
    set; the gap should shrink as n grows.

Averaged over a few seeds (data draw + weight init). Results cached to JSON;
two figures written to ``results/ite_train_size/plots/``.

Run from ``experiments/``::

    uv run python ite_train_size.py
    uv run python ite_train_size.py --seeds 3 --sizes 500 1000 2000 5000 10000 20000
"""

from __future__ import annotations

import argparse
import time

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

from paper_common import results_dir, save_json  # noqa: E402

from tramdag import CausalFlowDAG, ContinuousNode, I, OrdinalNode  # noqa: E402
from tramdag.simulations import ITEObservational  # noqa: E402
from tramdag.simulations.ite_observational import COLUMNS  # noqa: E402

SIZES = [500, 1000, 2000, 5000, 10000, 20000]
TEST_N = 5000
EPOCHS = 600


def make_spec() -> dict:
    """The notebook's all-CI S-learner spec (one joint CI per child node)."""
    return {"X1": ContinuousNode(), "X2": ContinuousNode(), "X3": ContinuousNode(),
            "Tr": OrdinalNode(levels=2, terms=[I("X1", "X2")]),
            "X5": ContinuousNode(terms=[I("Tr")]),
            "X6": ContinuousNode(terms=[I("X5")]),
            "Y":  ContinuousNode(terms=[I("Tr", "X1", "X2", "X3", "X5", "X6")])}


def evaluate(flow, test_obs, ite_true, true_ate) -> dict:
    """Per-individual ITE on the test set via abduction + do(Tr=0/1)."""
    u = flow.abduct(test_obs, seed=0)
    y1 = flow.sample(do={"Tr": 1.0}, u=u)["Y"].to_numpy()
    y0 = flow.sample(do={"Tr": 0.0}, u=u)["Y"].to_numpy()
    ite = y1 - y0
    return {"ite_pred": ite,
            "ate_pred": float(ite.mean()),
            "ate_err": float(ite.mean() - true_ate),
            "corr": float(np.corrcoef(ite, ite_true)[0, 1]),
            "mae": float(np.abs(ite - ite_true).mean())}


def run(sizes: list[int], n_seeds: int, scenario: int) -> dict:
    gen = ITEObservational(seed=123, scenario=scenario)
    true_ate = gen.true_ate(mc_n=400_000)["ate_true"]

    # one fixed test set, shared by every model
    test = gen.with_truth(TEST_N, seed_offset=999)
    test_obs = test[list(COLUMNS)]
    ite_true = test["ITE_true"].to_numpy()

    rows: list[dict] = []
    scatter: dict[int, np.ndarray] = {}    # seed-0 ITE predictions, for the grid
    for n in sizes:
        for s in range(n_seeds):
            train = gen.observational(n, seed_offset=10 + s)
            t0 = time.perf_counter()
            flow = CausalFlowDAG(make_spec(), seed=s)
            flow.fit(train, epochs=EPOCHS, learning_rate=1e-2, schedule="plateau",
                     plateau_patience=25, verbose=0)
            m = evaluate(flow, test_obs, ite_true, true_ate)
            row = {"n": n, "seed": s, "secs": time.perf_counter() - t0,
                   "train_nll": float(sum(flow.nll(train).values())),
                   "test_nll": float(sum(flow.nll(test_obs).values())),
                   **{k: v for k, v in m.items() if k != "ite_pred"}}
            rows.append(row)
            if s == 0:
                scatter[n] = m["ite_pred"]
            print(f"n={n:6d} seed={s}  ATE={row['ate_pred']:+.3f} "
                  f"(err {row['ate_err']:+.3f})  r={row['corr']:.3f}  "
                  f"MAE={row['mae']:.3f}  NLL train/test "
                  f"{row['train_nll']:.3f}/{row['test_nll']:.3f}  "
                  f"[{row['secs']:.0f}s]")
    return {"true_ate": true_ate, "scenario": scenario, "sizes": sizes,
            "n_seeds": n_seeds, "test_n": TEST_N, "epochs": EPOCHS,
            "rows": rows, "ite_true": ite_true, "scatter": scatter}


def _per_seed(rows, sizes, key):
    """List (per size) of the per-seed values — for scatter overlays."""
    return [np.array([r[key] for r in rows if r["n"] == n]) for n in sizes]


def _median(rows, sizes, key):
    """Median over seeds per size (robust to the small-n instability blow-up)."""
    return np.array([np.median([r[key] for r in rows if r["n"] == n])
                     for n in sizes])


def plot_scatter_grid(res, path) -> None:
    sizes, scatter, ite_true = res["sizes"], res["scatter"], res["ite_true"]
    rows_by = {(r["n"], r["seed"]): r for r in res["rows"]}
    lim = [ite_true.min(), ite_true.max()]
    ncol = 3
    nrow = int(np.ceil(len(sizes) / ncol))
    fig, axes = plt.subplots(nrow, ncol, figsize=(3.4 * ncol, 3.2 * nrow),
                             sharex=True, sharey=True)
    for ax, n in zip(axes.ravel(), sizes):
        ip = scatter[n]
        r = rows_by[(n, 0)]
        ax.plot(lim, lim, color="0.6", lw=1, ls="--")
        ax.scatter(ite_true, ip, s=5, alpha=0.25, color="#1b9e77")
        ax.set_title(f"n_train = {n}\nr={r['corr']:.3f}  ATE err={r['ate_err']:+.3f}",
                     fontsize=9)
        ax.set_xlabel("true ITE")
        ax.set_ylabel("predicted ITE")
    for ax in axes.ravel()[len(sizes):]:
        ax.set_visible(False)
    fig.suptitle("ITE recovery on the fixed 5k test set vs training size "
                 "(seed 0)", fontsize=11)
    fig.tight_layout()
    fig.savefig(path, dpi=100, bbox_inches="tight")
    plt.close(fig)


def _scatter_seeds(ax, sizes, per_seed, color, ylim):
    """Overlay per-seed points; those outside ylim are clipped to the top edge
    so an off-scale blow-up stays visible (as a clipped marker) without
    rescaling the axis."""
    n_out = 0
    for n, vals in zip(sizes, per_seed):
        clipped = np.clip(vals, *ylim)
        n_out += int((vals > ylim[1]).sum() + (vals < ylim[0]).sum())
        ax.scatter([n] * len(vals), clipped, s=18, color=color,
                   alpha=0.35, zorder=3)
    return n_out


def plot_curves(res, path) -> None:
    sizes, rows = res["sizes"], res["rows"]
    true_ate = res["true_ate"]
    # robust headline = median over seeds (the n=500 instability can blow a
    # single seed's ATE/MAE up by ~1e7; mean would be meaningless there)
    ate_med, corr_med, mae_med = (_median(rows, sizes, k)
                                  for k in ("ate_pred", "corr", "mae"))
    trn_med, tst_med = _median(rows, sizes, "train_nll"), _median(rows, sizes, "test_nll")

    fig, ax = plt.subplots(1, 3, figsize=(13, 4))
    # (1) ATE recovery — y-range pinned near the true ATE; blow-ups clip to edge
    ate_ylim = (true_ate - 0.35, true_ate + 0.35)
    ax[0].axhline(true_ate, color="0.5", ls="--", label="true ATE")
    n_out = _scatter_seeds(ax[0], sizes, _per_seed(rows, sizes, "ate_pred"),
                           "#d95f02", ate_ylim)
    ax[0].plot(sizes, np.clip(ate_med, *ate_ylim), marker="o", color="#d95f02",
               label="predicted ATE (median)")
    ax[0].set_ylim(*ate_ylim)
    ax[0].set(xscale="log", xlabel="training size n", ylabel="ATE",
              title="ATE recovery")
    if n_out:
        ax[0].annotate(f"{n_out} seed(s) off-scale\n(small-n instability)",
                       xy=(sizes[0], ate_ylim[0]), xytext=(0, 12),
                       textcoords="offset points", fontsize=7, color="#d95f02")
    ax[0].legend(fontsize=8, loc="lower right")
    # (2) ITE recovery — correlation (left) + MAE (right), median lines
    corr_ylim, mae_ylim = (-0.1, 1.0), (0.0, 0.45)
    _scatter_seeds(ax[1], sizes, _per_seed(rows, sizes, "corr"), "#1b9e77", corr_ylim)
    ax[1].plot(sizes, corr_med, marker="o", color="#1b9e77", label="corr (median)")
    ax[1].set_ylim(*corr_ylim)
    ax[1].set(xscale="log", xlabel="training size n", ylabel="ITE correlation",
              title="ITE recovery")
    ax1b = ax[1].twinx()
    _scatter_seeds(ax1b, sizes, _per_seed(rows, sizes, "mae"), "#7570b3", mae_ylim)
    ax1b.plot(sizes, np.clip(mae_med, *mae_ylim), marker="s", color="#7570b3",
              label="MAE (median)")
    ax1b.set_ylim(*mae_ylim)
    ax1b.set_ylabel("ITE MAE")
    h1, l1 = ax[1].get_legend_handles_labels()
    h2, l2 = ax1b.get_legend_handles_labels()
    ax[1].legend(h1 + h2, l1 + l2, fontsize=8, loc="center right")
    # (3) overfitting: train vs test NLL gap
    ax[2].plot(sizes, trn_med, marker="o", color="#1b9e77", label="train NLL")
    ax[2].plot(sizes, tst_med, marker="o", color="#d95f02", label="test NLL")
    ax[2].set(xscale="log", xlabel="training size n", ylabel="total NLL",
              title="overfitting: train vs test NLL")
    ax[2].legend(fontsize=8)
    fig.suptitle("ITE/ATE recovery vs training size (median over seeds; "
                 "points = seeds). Fixed 5k test set.", fontsize=10)
    fig.tight_layout()
    fig.savefig(path, dpi=100, bbox_inches="tight")
    plt.close(fig)


def main(argv: list[str] | None = None) -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--sizes", type=int, nargs="+", default=SIZES)
    p.add_argument("--seeds", type=int, default=3)
    p.add_argument("--scenario", type=int, default=1, choices=(1, 2, 3, 4))
    args = p.parse_args(argv)

    out = results_dir("ite_train_size")
    res = run(args.sizes, args.seeds, args.scenario)

    # JSON-friendly copy (drop the big arrays)
    save_json(out / "results.json",
              {k: v for k, v in res.items() if k not in ("scatter", "ite_true")})
    plot_scatter_grid(res, out / "plots" / "ite_scatter_grid.png")
    plot_curves(res, out / "plots" / "ate_ite_curves.png")
    print(f"\nwrote {out}/results.json and 2 plots under {out}/plots/")


if __name__ == "__main__":
    main()
