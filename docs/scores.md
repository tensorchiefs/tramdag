# Per-observation scores and the effect-modifier scan

A treatment effect is often not the same for every patient. A covariate that
changes the effect is an effect modifier. A `VC` term models an effect
$\beta(x)$ that changes with the modifiers, but you must tell it which
covariates they are. The effect-modifier scan finds candidates for that list.
It uses a simple model that takes seconds to fit, before you fit the flexible
one. [varying-coefficients.md](varying-coefficients.md) covers the `VC` term
itself.

## The idea

1. **Fit a simple model.** It assumes one constant effect $\beta$ for
   everyone. Every parent enters as a linear shift (`LS`), and
   `fit_classical` fits this all-`LS` model in seconds.
2. **Ask each row which way $\beta$ should move.** The score of row $i$ is
   $\psi_i = \partial \ell_i / \partial \beta$, the slope of that row's
   log-likelihood $\ell_i$. A positive score means the row fits better with a
   larger $\beta$. A negative score means it fits better with a smaller one.
3. **Check that the answers cancel in every group.** At the fitted $\beta$
   the scores sum to about zero, because $\beta$ is the best compromise. If
   the effect is really constant, the scores of any group of rows also sum to
   about zero, apart from noise.
4. **Sort and add up.** Sort the rows by a candidate covariate, for example
   age. The sum of the first $j$ scores is the score sum of the group "age up
   to the $j$-th value". So the running sum checks all these groups at once.

Plotted over the sorted rows, the running sum forms a path. Suppose young
patients all ask for a larger $\beta$ and old patients for a smaller one.
The path then climbs first and falls back later. A large swing of the path
along a covariate means that one constant $\beta$ hides a pattern along that
covariate.

## Reading the running sum

Section 2 of
[`notebooks/varying_coefficients.py`](../notebooks/varying_coefficients.py)
draws the running sum for three covariates. The true effect changes with `X2`
and `X3` and not with `X1`. The figure has two panels with the same effect.
In the left panel `X1` has a quadratic effect on the outcome, and in the
right panel a linear one.

The figure plots the running sum after a scaling, and the grey band spans
$\pm 1.358$ on that scale ([The statistic](#the-statistic) explains both). If
a path leaves the band, the scan flags its covariate.

- The paths of `X2` and `X3` swing far out of the band and come back. They
  flag in both panels.
- In the right panel the path of `X1` stays in the band. The simple model
  describes the linear effect of `X1` correctly, and `X1` does not flag.
- In the left panel the path of `X1` swings out as well. The simple model
  cannot describe the quadratic effect, and `X1` flags although it does not
  change the treatment effect. The section
  [Read it as screening](#read-it-as-screening) explains why.

## The statistic

The scan is the structural-change test [@zeileis2007fluctuation] behind
model-based recursive partitioning [@zeileis2008mob; @dandl2024forest]. That
method is a tree that splits the data where the coefficients
of a model change. The scan applies its test to the treatment coefficient of
a TRAM-DAG.

For one candidate, the scan sorts the $n$ rows by that candidate. It then
forms the scaled running sum, also called a CUSUM (cumulative sum):

$$
B_j = \frac{1}{\mathrm{sd}(\psi)\sqrt{n}} \sum_{i \le j} \psi_{(i)},
\qquad j = 1, \dots, n,
$$

with $\psi_{(i)}$ the score of the $i$-th row in the sorted order. The
statistic `stat` is $\max_j |B_j|$, the largest distance of the path from
zero.

- The path ends near 0, because the scores sum to about zero. It starts near
  0, because one score divided by $\sqrt{n}$ is small.
- The scaling by $\mathrm{sd}(\psi)\sqrt{n}$ makes the distribution of
  `stat` approximately independent of the sample size and of the size of
  the scores.
- With a constant effect, $B$ behaves approximately like a Brownian bridge: a
  random path that starts and ends at 0.
- The largest distance of a Brownian bridge from zero has a known
  distribution, the Kolmogorov distribution. It gives the `p_value` without
  simulation.
- A `stat` above 1.358, the 5 % critical value, sets `flag`.

The p-value is approximate. The scan uses the raw treatment score. The full
test [@zeileis2007fluctuation] removes its correlation with the scores of the
other fitted coefficients, and the scan does not.

The error has a direction. For a candidate that is also an `LS` parent of
the node, the fitted `LS` weight already removes part of the pattern. The
scan then flags too rarely. In 100 simulated data sets of 2000 rows, from
the notebook's linear model with the constant effect $\beta = -1$, each of
the three parents flagged in 1 % of them or fewer. An independent covariate
that is not a parent flagged in 4 %, close to the nominal 5 %. Read
the p-value as a guide, not as an exact error rate.

## Read it as screening

A flag means that one constant $\beta$ does not fit all rows along the
covariate. Two causes give this:

- The effect really changes with the covariate.
- The simple model describes the covariate's own effect on the outcome, the
  prognostic part, wrongly.

The second cause works through the score. The treatment score of row $i$ is
$\psi_i = r_i \, t_i$. Here $t_i$ is the treatment of the row, and
$r_i = \partial \ell_i / \partial s_i$ is the derivative of $\ell_i$ with
respect to the row's total shift $s_i$. The factor $r_i$
acts like a residual. If the simple model gets the prognostic effect
of `X1` wrong, $r_i$ follows a pattern in `X1`. The treatment score takes on
that pattern, so the path of `X1` swings.

So a flag means "look here". What you find is a modifier or a
misspecification, and both are worth knowing.

Two cases make the p-value less reliable:

- **Few distinct values.** A binary or three-level candidate gives many tied
  rows. Inside a tie the order is arbitrary, and the path depends on it.
  Read the p-value as a ranking, not as an exact test.
- **No converged fit.** The scores sum to zero only at the optimum. Scan an
  all-`LS` model whose `fit_classical` report says `converged`. The scan
  also takes the $\beta_0$ of a `VC` term, but a `VC` model stopped at its
  best-validation weights is not at the optimum. Read such a scan with more
  care.

## Using it

`flow.effect_modifier_scan(df, node, t=)` runs the scan for the treatment
`t` of `node`. It returns one row per candidate with `stat`, `p_value`,
`crit_5pct` and `flag`, sorted by `stat`.

- `candidates=` lists the covariates to scan. By default it is every column
  of `df` except `node` and `t`. A candidate does not have to be a parent.
- For a binary ordinal `LS` treatment, `t` resolves to the identified
  contrast of level 1 against level 0. For a `VC` treatment it resolves to
  $\beta_0$, the constant part of $\beta(x)$.
- `column=` picks one score column by name, for example one level contrast
  `"t[2]"` of a treatment with more than two levels.

`flow.scores(df, node)` returns the scores themselves: one row for each row
of `df`, and one column per interpretable shift coefficient of the node.

- A continuous `LS` parent gives one column, named after the parent.
- An ordinal `LS` parent gives one column per level of its one-hot
  encoding, named `"t[0]"`, `"t[1]"` and so on. A row's score goes to the
  column of the level that the row has.
- A `VC` term gives one column for its $\beta_0$, named after the
  treatment.
- A `CS` term is a network with no single coefficient, so it has no column.

The scores also serve influence analyses and robust standard errors.

## How the scores are computed

Every shift coefficient $\gamma$ adds $\gamma z$ to the node's total shift
$s$. Here $z$ is the value of a parent, or one column of its one-hot
encoding. The chain rule therefore gives

$$
\frac{\partial \ell_i}{\partial \gamma}
= \frac{\partial \ell_i}{\partial s_i}\, z_i = r_i \, z_i,
$$

so one derivative $r_i$ per row serves every coefficient. The derivative has
a closed form for both node kinds, and the computation uses no autograd. Below,
$h$ is the fitted monotone transformation and $\sigma$ the logistic
function.

- **Continuous node.** For the node value $y$ the latent value is
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
