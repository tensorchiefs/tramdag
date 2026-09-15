# The model

A TRAM-DAG is one triangular normalizing flow whose sparsity is a causal DAG.
This page states the model in the notation of the paper and maps each part of
it onto the constructors in `tramdag.spec`. For the symbols see
[notation.md](notation.md); to read a model after it is fitted see
[interpretation.md](interpretation.md); for a worked fit see the
[demo notebook](../notebooks/demo_tram_dag_colab.py).

## The transformation function

The variables have a causal ordering. For each variable a TRAM-DAG fits a
monotone transformation function $h$. The function maps the observed value to
a latent scale, conditional on the parents of that variable:

$$
\begin{aligned}
u_1 &= h(x_1) \\
u_2 &= h(x_2 \mid x_1)\\
u_3 &= h(x_3 \mid x_1, x_2) \\
u_p &= h(x_p \mid x_1, x_2, \dots, x_{p-1})
\end{aligned}
$$

This is the observed-to-latent direction. It is Eq. 2 of the paper,
$F_{X\mid\mathrm{pa}}(x)=F_U\big(h(x\mid\mathrm{pa})\big)$, and in code it is
`u = h(x) + shift`.

The direction matters for cost. Training evaluates $h$ directly to score the
likelihood, which is cheap. Sampling runs the inverse
$x_i = h^{-1}(u_i \mid \mathrm{pa})$. That inverse has no closed form, so
bracketed bisection solves it, and sampling is the more expensive direction.

Together the $h$'s form one triangular flow. Each variable depends only on its
causal parents $\mathrm{pa}(x_i)$, which is a subset of its predecessors. The
Jacobian sparsity of the flow is therefore the DAG itself.

The latents $u_1,\dots,u_p$ are standard logistic. That choice is what makes
the fitted parameters interpretable, because a shift on the latent scale is
then a log-odds ratio.

## The components

The transformation decomposes additively on the latent scale, which is what
keeps the interpretation valid. Each node's $h$ is

$$
u_i \;=\; h(x_i \mid \mathrm{pa}(x_i)) \;=\;
\underbrace{h_{\boldsymbol{\vartheta}}(x_i)}_{\text{intercept}}
\;+\; \underbrace{\textstyle\sum_j \beta_{ij}\, x_j}_{\text{linear shifts}}
\;+\; \underbrace{\textstyle\sum_k g_{ik}(x_k)}_{\text{complex shifts}}
\;+\; \underbrace{(\beta_0 + b_{\boldsymbol{\Theta}}(x_{\text{mod}}))\, x_t}_{\text{varying coefficient}} ,
$$

and every causal parent enters through exactly one term. Take $x_5$ with
parents $\mathrm{pa}(x_5) = \{x_1, x_2, x_4\}$ as the example.

- **Simple intercept, SI.** $h_{\boldsymbol{\vartheta}}(x_5)$ has constant
  parameters. It is a flexible monotone baseline, a Bernstein polynomial by
  default, and it is the same for every observation.
- **Complex intercept, CI.** The parameters $\boldsymbol{\vartheta}$ are
  themselves a function of some parents. The whole transformation bends with
  the parent, which permits interactions that an additive shift cannot express.
- **Linear shift, LS.** $\beta_{51} x_1 + \beta_{52} x_2$. One interpretable
  number per parent.
- **Complex shift, CS.** $g(x_4)$, an unrestricted network of the parent. It
  stays additive on the latent scale.
- **Varying coefficient, VC.** $(\beta_0 + b_{\boldsymbol{\Theta}}(x_{\text{mod}}))\, x_t$:
  a treatment $x_t$ whose effect $\beta$ varies with modifier covariates
  through a small penalized network. [varying-coefficients.md](varying-coefficients.md)
  is its guide.
- **Function shift, Fn.** $f(x_{\text{pa}})$ for a user-supplied function or
  `nn.Module`, for a one-off term.

Sampling has to invert only the intercept, because the shifts move to the
other side:

$$
x_5 = h^{-1}(u_5 \mid x_1, x_2, x_4) = h_{\boldsymbol{\vartheta}}^{-1}\Big(u_5
- \underbrace{\beta_{51} x_1 + \beta_{52} x_2}_{\text{LS}}
- \underbrace{g(x_4)}_{\text{CS}}\Big).
$$

## The components in code

Each node declares its transformation as an additive formula of terms. The
formula is the node's first argument, a `+` sum of terms. Each constructor
names the parents its term depends on. The paper's symbols `I`, `LS`, `CS`, `VC`, `Fn` are the classes
`Intercept`, `LinearShift`, `ComplexShift`, `VaryingCoefficient`, `FnShift`.

| formula for a continuous node $x_3$ | $u_3 = h(x_3 \mid \mathrm{pa})$ |
|---|---|
| nothing, or `I()` | $h_{\vartheta}(x_3)$, the simple intercept |
| `LS("X1")` | $h_{\vartheta}(x_3) + \beta x_1$ |
| `I("X1")` | $h_{\vartheta(x_1)}(x_3)$, the complex intercept |
| `CS("X1")` | $h_{\vartheta}(x_3) + g_1(x_1)$ |
| `LS("X1") + CS("X2")` | $h_{\vartheta}(x_3) + \beta x_1 + g_2(x_2)$ |
| `CS("X1", "X2")` | $h_{\vartheta}(x_3) + g_{12}(x_1, x_2)$, one joint network |
| `CS("X1") + CS("X2")` | $h_{\vartheta}(x_3) + g_1(x_1) + g_2(x_2)$, two additive networks |
| `I("X1", "X2")` | $h_{\vartheta(x_1, x_2)}(x_3)$, one joint network |
| `I("X1", "X2", allow_interaction=False)` | $h_{\vartheta(x_1) + \vartheta(x_2)}(x_3)$, one network per parent, summed in coefficient space |
| `CS("X1") + VC("X2", t="T")` | $h_{\vartheta}(x_3) + g_1(x_1) + \beta(x_2)\, x_T$ |

Three rules follow from the model.

- **Exactly one intercept, first.** A formula written without one gets `I()`
  prepended. `I()` without parents is the simple intercept, `I(...)` with
  parents the complex one; `SI()` and `CI(...)` spell that out with the arity
  checked.
- **Joint versus additive is argument grouping.** Several parents in one term
  form one network over all of them, an interaction; the same parents in
  separate terms act additively. For the intercept the grouping is said with
  `allow_interaction`, because a node takes at most one intercept with
  parents.
- **Every parent enters through exactly one edge-owning term.** The VC
  modifiers are the one exception: `CS("X2") + VC("X2", t="T")` is the
  intended pattern, where $x_2$ acts prognostically through the shift and
  modifies the treatment effect.

### The three knobs on a term

**`transform=` on `I`** picks the class of $h_{\boldsymbol{\vartheta}}$ for a
continuous node:

- `"bernstein"`, the default: a Bernstein polynomial with `n_coeffs=20`
  unconstrained coefficients whose tails extrapolate along the boundary
  slope;
- `"spline"`: a monotone rational-quadratic spline with `bins=8`, whose tail
  slope is fixed ([zuko-upstream.md](zuko-upstream.md) explains the
  consequence);
- `"affine"`: location and scale only, so the node-conditional is a logistic
  GLM.

Extra keyword arguments of `I` pass straight to the transform class,
`I(transform="spline", bins=16)`. Each transform pre-scales the data from the
training `range_q` and $1 - $`range_q` quantiles onto a fixed domain;
`range_q` is an intercept option, default 0.05, and `range_q=0` uses the
minimum and maximum. Ordinal nodes have no transform to pick.

**`input_transform=` on `I`, `CS`, `VC`** transforms that term's continuous
network inputs, with statistics frozen at the first fit: `"minmax"`,
`"standardize"`, or a callable `fn(x, train)`. `LS` and the `VC` treatment
stay raw, so their coefficients keep their units.

**`units=`, `activation=`, `batch_norm=` on `I`, `CS`, `VC`** size the
term's network. [code-map.md](code-map.md) lists every default.

## Ordinal nodes

An ordinal node has no monotone transform to choose. Its intercept is the
cutpoint vector of an ordered logit,

$$
P(Y \le k \mid \mathrm{pa}) = \sigma(\vartheta_k - s(\mathrm{pa})),
$$

with increasing cutpoints $\vartheta$ and $s$ the total shift. The shift is
subtracted where a continuous node adds it; [notation.md](notation.md) states
the convention and [interpretation.md](interpretation.md) works through what
it does to the sign of a coefficient.

The log-probability of an observed level is the difference of two sigmoids,
computed in log space with `logsigmoid` and a stable `log(1 - exp(x))` and
taking per element the better-conditioned side. The direct difference of two
sigmoids has exactly zero gradient once they saturate in float32, and a node
that starts there never recovers.

Parents enter every network as features: a continuous parent raw, in one
column; an ordinal parent one-hot, in one column per level. Abduction is exact
for continuous nodes and truncated-logistic for ordinal ones, so
`flow.sample(u=flow.abduct(df))` reproduces `df` exactly, level-exactly for
ordinal nodes.

## What the model cannot do

- The latent distribution is fixed to standard logistic. It is not learned.
- A `VC` treatment is continuous or a binary ordinal node; a multi-level
  ordinal treatment is not supported. Propensity centering needs a binary
  ordinal treatment.
- The DAG is an input. `tramdag` fits the mechanisms of a graph you supply, and
  does not discover the graph.
- Identification is the usual causal one. No hidden confounding, and the graph
  has to be right.
