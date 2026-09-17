# Per-observation scores and the effect-modifier scan

`flow.scores(df, node)` returns the score
$\psi_i = \partial \ell_i / \partial \theta$ of each observation for the
node's interpretable shift coefficients: every `LS` weight and every `VC`
term's $\beta_0$. An `LS` weight gives one column per continuous parent and
one per one-hot level of an ordinal parent; the $\beta_0$ column is named
after the treatment.

The computation is analytic, with no autograd. Shifts enter the latent
additively, so $\partial \ell_i / \partial \beta = (\partial \ell_i /
\partial s_i)\, x_i$ with $\partial \ell_i / \partial s$ in closed form. The
function is a pure read-out that touches neither fitting nor sampling. At a
fitted MLE each column sums to about zero, which the tests pin together with
a float64 finite-difference check.

## The effect-modifier scan

`flow.effect_modifier_scan(df, node, t=)` is the cheapest effect-modifier
detector available. It applies the structural-change logic of model-based
recursive partitioning (Zeileis & Hornik; Dandl et al. 2024) to the
treatment-coefficient scores of a cheap all-`LS` fit, which `fit_classical`
delivers in seconds.

For each candidate covariate the scan orders the scores by that covariate and
forms the scaled cumulative sum $B_j = \sum_{i \le j} \psi_{(i)} /
(\mathrm{sd}(\psi)\sqrt{n})$. Under a stable coefficient $B$ is a Brownian
bridge, so $\sup |B|$ has the Kolmogorov distribution, with the 5 % critical
value 1.358. A coefficient that varies with the covariate makes the ordered
scores drift. The result is one row per candidate with the statistic, its
p-value, the critical value and a flag; the flagged covariates are the
shortlist of `VC` modifiers to measure
([varying-coefficients.md](varying-coefficients.md)).

**Read it as screening, not as a test of effect modification.** The statistic
measures how unstable the cheap model is along a covariate, and instability
has two sources: a coefficient that genuinely varies, and a prognostic part
the cheap model gets wrong. A prognostic-only covariate with a linear effect
stays unflagged; give it a quadratic effect and it flags as strongly as the
true modifiers. A flag therefore means "look here", and what you find is a
modifier or a misspecification, both worth knowing.
[`notebooks/varying_coefficients.py`](../notebooks/varying_coefficients.py)
shows both cases side by side.

Details: for a binary ordinal `LS` treatment, `t` resolves to the identified
level-1-against-0 contrast, and for a `VC` treatment to $\beta_0$. Candidates
default to every column of `df` except the node and `t`, and need not be
parents. For few-level, heavily tied candidates the ordering is only partial,
so read the scan as a ranking diagnostic rather than an exact-size test. The
scores also serve influence-function analyses and robust standard errors.
