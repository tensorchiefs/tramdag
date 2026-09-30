# Per-observation scores and the effect-modifier scan

A treatment effect is often not the same for every patient. A `VC` term
models an effect $\beta(x)$ that changes with covariates, but you must tell
it which covariates. The effect-modifier scan finds candidates for that list.
It uses a simple model that takes seconds to fit, before you fit the flexible
one. [varying-coefficients.md](varying-coefficients.md) covers the `VC` term
itself.

## The idea

1. **Fit a simple model.** It assumes one constant effect $\beta$ for
   everyone: an all-`LS` model, which `fit_classical` fits in seconds.
2. **Ask each row which way $\beta$ should move.** The score of row $i$ is
   $\psi_i = \partial \ell_i / \partial \beta$, the slope of that row's
   log-likelihood $\ell_i$. A positive score means the row fits better with a
   larger $\beta$. A negative score means it fits better with a smaller one.
3. **Check that the answers cancel everywhere.** At the fitted $\beta$ the
   scores sum to about zero, because $\beta$ is the best compromise. If the
   effect is really constant, they also cancel within every group of rows.
   Now sort the rows by a candidate covariate, for example age, and add up
   the scores in that order. Suppose young patients all ask for a larger
   $\beta$ and old patients for a smaller one. The running sum then climbs
   first and falls back later.

A large swing of the running sum along a covariate means that one constant
$\beta$ hides a pattern along that covariate.

## Reading the running sum

Section 2 of
[`notebooks/varying_coefficients.py`](../notebooks/varying_coefficients.py)
draws the running sum for three covariates. The true effect changes with `X2`
and `X3` and not with `X1`:

- The paths of `X2` and `X3` swing far out of the grey band and come back.
  They flag.
- The path of `X1` stays in the band when the simple model describes `X1`
  correctly. It does not flag.
- When `X1` has a quadratic effect on the outcome, which the simple model
  cannot describe, the path of `X1` swings out as well. It flags although
  `X1` does not change the effect. The section
  [Read it as screening](#read-it-as-screening) explains this.

The grey band marks the 5 % critical value. A path that leaves the band
flags.

## The statistic

For one candidate the scan sorts the $n$ rows by that candidate and forms the
scaled running sum, also called a CUSUM (cumulative sum)

$$
B_j = \frac{1}{\mathrm{sd}(\psi)\sqrt{n}} \sum_{i \le j} \psi_{(i)},
\qquad j = 1, \dots, n,
$$

with $\psi_{(i)}$ the score of the $i$-th row in the sorted order. The
statistic `stat` is $\sup_j |B_j|$, the largest distance of the path from
zero.

- The path starts and ends near 0, because the scores sum to about zero.
- The scaling by $\mathrm{sd}(\psi)\sqrt{n}$ makes the distribution of
  `stat` independent of the sample size and of the size of the scores.
- With a constant effect, $B$ behaves approximately like a Brownian bridge: a
  random path that starts and ends at 0. So `stat` approximately follows the
  Kolmogorov distribution, which gives the `p_value` without simulation.
- The approximation is not exact. The scan uses the raw treatment score and
  does not remove its correlation with the scores of the other fitted
  coefficients, as the full test of Zeileis & Hornik does. Read the p-value
  as a guide, not as an exact size.
- A `stat` above 1.358, the 5 % critical value, sets `flag`.

This is the structural-change test of model-based recursive partitioning
(Zeileis & Hornik; Dandl et al. 2024), applied to the treatment coefficient
of a TRAM-DAG.

## Read it as screening

A flag means that the simple model is unstable along the covariate. Two
causes give this:

- the effect really changes with the covariate, or
- the simple model describes the covariate's own effect on the outcome
  (the prognostic part) wrongly.

A covariate with a linear prognostic effect stays unflagged. Give it a
quadratic prognostic effect and it flags as strongly as the true modifiers.
So a flag means "look here". What you find is a modifier or a
misspecification, and both are worth knowing.

Two cases make the p-value rougher still:

- **Few distinct values.** A binary or three-level candidate sorts the rows
  only partly, because many rows tie. Read the p-value as a ranking, not as
  an exact test.
- **No converged fit.** The scores sum to zero only at the optimum. Scan an
  all-`LS` model that `fit_classical` reports as converged.

## Using it

`flow.effect_modifier_scan(df, node, t=)` runs the scan for the treatment
`t` of `node`. It returns one row per candidate with `stat`, `p_value`,
`crit_5pct` and `flag`, sorted by `stat`.

- `candidates=` lists the covariates to scan. By default it is every column
  of `df` except `node` and `t`. A candidate does not have to be a parent.
- For a binary ordinal `LS` treatment, `t` resolves to the identified
  contrast of level 1 against level 0. For a `VC` treatment it resolves to
  $\beta_0$.
- `column=` picks one score column by name, for example one level contrast
  `"t[2]"` of a treatment with more than two levels.

`flow.scores(df, node)` returns the scores themselves: one row per
observation and one column per interpretable shift coefficient of the node.

- A continuous `LS` parent gives one column, named after the parent.
- An ordinal `LS` parent gives one column per level of its one-hot
  encoding, named `"t[0]"`, `"t[1]"` and so on. A row's score goes to the
  column of the level that the row has.
- A `VC` term gives one column for its $\beta_0$, named after the
  treatment.
- A `CS` term is a network with no single coefficient, so it has no column.

The scores also serve influence analyses and robust standard errors.

## How the scores are computed

Every shift coefficient adds $\beta x_j$ to the node's total shift $s$. Here
$x_j$ is the value of a parent, or one column of its one-hot encoding. The
chain rule therefore gives

$$
\frac{\partial \ell_i}{\partial \beta}
= \frac{\partial \ell_i}{\partial s_i}\, x_{ij} ,
$$

so one derivative per row serves every coefficient. The derivative has a
closed form for both node kinds, and the computation uses no autograd.

- **Continuous node.** For the node value $y$ the latent is
  $u = h(y) + s$ and $\ell = \log f(u) + \log h'(y)$, with $f$ the
  standard-logistic density. So $\partial \ell / \partial s = 1 - 2\sigma(u)$.
- **Ordinal node.** $P(Y = y) = \sigma(b) - \sigma(a)$ with the shifted
  cutpoints $a = \vartheta_{y-1} - s$ and $b = \vartheta_y - s$, as in
  [model.md](model.md#ordinal-nodes). So
  $\partial \ell / \partial s = \bigl(\sigma'(a) - \sigma'(b)\bigr) /
  \bigl(\sigma(b) - \sigma(a)\bigr)$, with $\sigma' = \sigma(1 - \sigma)$.
  For the first and last class the outer cutpoint is $\mp\infty$ and its
  $\sigma'$ is zero.

The function only reads the fitted model. It changes neither fitting nor
sampling. The tests check that each column sums to about zero at a fitted
optimum, and that the scores agree with float64 finite differences.
