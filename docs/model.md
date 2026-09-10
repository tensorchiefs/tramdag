# The model

A TRAM-DAG is one triangular normalizing flow whose sparsity is a causal DAG.
This page states the model in the notation of the paper, and maps each part of
it onto the constructors in `tramdag.spec`. For the symbols themselves see
[notation.md](notation.md). To read a model after it is fitted see
[interpretation.md](interpretation.md).

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

## The four components

The transformation decomposes additively on the latent scale, which is what
keeps the interpretation valid. Each node's $h$ is

$$
u_i \;=\; h(x_i \mid \mathrm{pa}(x_i)) \;=\;
\underbrace{h_{\boldsymbol{\vartheta}}(x_i)}_{\text{intercept}}
\;+\; \underbrace{\textstyle\sum_j \beta_{ij}\, x_j}_{\text{linear shifts}}
\;+\; \underbrace{\textstyle\sum_k g_{ik}(x_k)}_{\text{complex shifts}} ,
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

Sampling has to invert only the intercept, because the shifts move to the
other side:

$$
x_5 = h^{-1}(u_5 \mid x_1, x_2, x_4) = h_{\boldsymbol{\vartheta}}^{-1}\Big(u_5
- \underbrace{\beta_{51} x_1 + \beta_{52} x_2}_{\text{LS}}
- \underbrace{g(x_4)}_{\text{CS}}\Big).
$$

## The components in code

Each node declares its transformation as an additive formula of terms. The
formula is the node's first argument, written as a list or as a `+` sum. Each
constructor names the parents its term depends on.

| paper component | `tramdag` |
|---|---|
| SI, baseline $h_{\boldsymbol{\vartheta}}(x_i)$ with constant $\boldsymbol{\vartheta}$ | automatic: every node owns a monotone transform. Without an intercept term its $\boldsymbol{\vartheta}$ is a free parameter vector |
| CI, $\boldsymbol{\vartheta}$ depends on parents | `I("X1")`. Several parents in one `I(...)` feed one joint network, which is how interactions arise |
| LS, $\beta_{ij} x_j$ | `LS("X1")`, a single weight and no bias |
| CS, $g_{ik}(x_k)$ | `CS("X1")`, an additive network |

`I(...)` dispatches on its arguments. Without parents it is a simple
intercept, with parents a complex one. `SI()` and `CI(...)` spell that out and
check the arity.

The formulas for each combination of terms, and the difference between a joint
and an additive grouping, are tabulated in the
[README](https://github.com/tensorchiefs/tramdag#the-model-in-detail-spec--math--networks).

## Ordinal nodes

An ordinal node has no monotone transform to choose. Its intercept is the
cutpoint vector of an ordered logit. The model is
$P(Y \le k) = \mathrm{sigmoid}(\theta_k - s)$, where $\theta$ holds the
increasing cutpoints and $s$ is the total shift.

Note the sign. A continuous node adds its shift on the latent scale, and an
ordinal node subtracts it. Both follow the original TRAM-DAG conventions, and
the test suite pins them. A parent that raises a continuous child therefore
gets a negative weight, which
[interpretation.md](interpretation.md) works through.

## What the model cannot do

- The latent distribution is fixed to standard logistic. It is not learned.
- Every parent must enter through exactly one edge-owning term. The one
  exception is a varying-coefficient modifier, which may also act
  prognostically through another term.
- The DAG is an input. `tramdag` fits the mechanisms of a graph you supply, and
  does not discover the graph.
- Identification is the usual causal one. No hidden confounding, and the graph
  has to be right.
