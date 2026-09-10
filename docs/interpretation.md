# Reading a fitted TRAM-DAG

A TRAM-DAG is a normalizing flow, but it is built so that parts of it stay
readable. This page collects the read-outs and says what each number means.
For the model itself see [model.md](model.md), and for the symbols see
[notation.md](notation.md).

Every claim on this page is checked somewhere that runs. The links say where,
so nothing here is a number typed into prose.

## Which term owns which edge

`flow.to_matrix()` gives the labelled adjacency matrix of the model. Rows are
parents and columns are children. A cell holds the term tag: `LS`, `CS`, `CI`,
`VC` for a treatment edge, or `VCm` for a varying-coefficient modifier. An
empty cell means no edge. A multi-parent term carries its parent group as a
suffix, and several terms in one cell join with `+`.

This is the meta-adjacency view of the paper. It is also the fastest way to
confirm that the model you wrote is the model you meant.
`plot_dag(spec)` draws the same thing, with one edge style per term.

Exercised in the [varying-coefficients notebook](../notebooks/varying_coefficients.py).

## A linear shift is a log-odds ratio

`flow.ls_coefficients()` gives `{node: {parent: weights}}` for the `LS` terms
only. A `CS` or `VC` shift is a network and has no single weight, so those are
skipped.

Two things decide how to read a weight.

**The sign follows the latent-scale convention.** A continuous node adds its
shift, so a parent that raises the child gets a *negative* weight. An ordinal
node subtracts its shift, so the sign is the one you expect there. This is
not a quirk of the implementation. It is the convention of the original
TRAM-DAG work, and the test suite pins it.

**The scale is the latent scale, not the data scale.** For a continuous node
the weights carry the transform's scaling. A single weight is therefore not
comparable to a regression coefficient. Their *ratio* is comparable, because
the scaling cancels. An ordinal node has no such scaling, so $\exp(\beta)$ is
an odds ratio directly.

**An ordinal parent gives one weight per level**, from its one-hot encoding.
Those weights are identified only up to a common constant. The one-hot columns
sum to one in every row. Adding a constant to all of them, and subtracting it
from the intercept, leaves the likelihood unchanged. Read them as
differences.
`w[k] - w[0]` is the level-k-against-level-0 log-odds ratio, and it is the
column that `flow.design_matrix(df, node, drop_first=True)` drops.

Checked in [`notebooks/demo_tram_dag_colab.py`](../notebooks/demo_tram_dag_colab.py),
which recovers a coefficient ratio from a known process, and in
[`notebooks/classical_fit_tram_dag.py`](../notebooks/classical_fit_tram_dag.py),
which compares the weights against `statsmodels` and R. The odds-ratio
reading is pinned in continuous integration by
`experiments/paper/triangle_mixed.py`.

## Ordinal nodes: cutpoints and the interventional PMF

An ordinal node's intercept is its cutpoint vector, not a transform. The model
is

$$
P(Y \le k \mid \mathrm{pa}) = \mathrm{sigmoid}(\theta_k - s(\mathrm{pa})),
$$

with increasing cutpoints $\theta$ and $s$ the total shift.
`tramdag.transforms.ordinal_cutpoints` turns the unconstrained parameters into
those increasing cutpoints, which is how you read a fitted baseline.

`flow.pmf(df, node, do=...)` gives the class probabilities per row, in closed
form. It needs no sampling, and `do=` makes it interventional by overriding
columns before the parents are read. That combination is what lets a
treatment-effect question be answered exactly rather than by Monte Carlo.

Exercised in the [classical-fitting notebook](../notebooks/classical_fit_tram_dag.py),
and under `do=` in `experiments/misc/validate_ls.py`, which is checked against
committed ground truth on every run.

## Continuous nodes: density and one fitted shift

`flow.density(df, node, grid, do=...)` is the continuous counterpart of
`pmf`. For every row it evaluates $p(\text{node} = g \mid \mathrm{pa})$ at
each grid value, in closed form from the transform. Use it to look at a fitted
conditional without sampling it.

`flow.shift_curve(node, parent, grid)` evaluates one fitted shift term along a
grid of parent values. Use it to plot a `CS` term against the function it was
supposed to learn. It goes through the term's own evaluation, so a custom term
and an `Fn` term work too.

A caution on both: a `CS` term is identified only up to a constant, because
that constant can move into the intercept. Compare a fitted curve to the truth
after mean-centring both, not point by point.

## The price of a complex intercept

`I(...)` with parents is the flexible end of the family. The parents control
the transform parameters, so the node can bend in ways no additive shift can
express. The cost is that there is no single coefficient to read.

That is a real trade-off, and it is measurable. `experiments/paper/` fits the
same process twice, once with an `LS` edge and once with a `CS` edge. It
compares both against the known truth, and
[paper-replication.md](paper-replication.md) reports the result. The short
version has two halves. Where the true edge is linear, the interpretable model
gives up very little likelihood. Where the edge is curved, the flexible model
wins, and the interpretable one absorbs the curvature into a biased single
number.

A middle option exists. `I("a", "b", allow_interaction=False)` builds one
network per parent and sums their parameter vectors, so the parents act
additively on the transform. `flow.intercept_contributions(df, node)` then
decomposes that sum into mean-centred per-term parts, using the usual additive
model convention, so the parts become comparable. This is a post-hoc read-out
and changes nothing about the fitted model.

Worked through in the [intercept notebook](../notebooks/additive_vs_joint_ci.py).

## An effect that varies with a covariate

`VC(*modifiers, t=...)` fits $\beta(\text{modifiers}) \cdot x_t$, and
`flow.varying_coef(df, node)` reads the fitted $\beta$ back. It is
deterministic, needs no abduction, and for a binary treatment it equals the
abduction difference.

`flow.scores(df, node)` and `flow.effect_modifier_scan(df, node, t=...)` come
before that decision rather than after it. They rank candidate modifiers by
how much the treatment coefficient's per-observation scores drift when the
rows are ordered by each candidate. A cheap all-`LS` fit is enough, so the
shortlist is measured rather than guessed.

See [varying-coefficients.md](varying-coefficients.md) and
[scores.md](scores.md), and
[`notebooks/varying_coefficients.py`](../notebooks/varying_coefficients.py).
