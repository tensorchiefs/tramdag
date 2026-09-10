# Upstream candidates for zuko

This document lists what tramdag works around in
[zuko](https://zuko.readthedocs.io/stable/) (1.6.0). The list is ranked by
value against effort, as candidate upstream PRs or issues. tramdag uses zuko
in one place only. It uses these three transforms from `zuko.transforms`:

- `BernsteinTransform`
- `MonotonicRQSTransform`
- `MonotonicAffineTransform`

The three wrappers in `src/tramdag/transforms.py` hold them. The flow
machinery, the base distribution and the DAG structure are tramdag's own. The
suggested order of attack is 4 → 1 → 3 → 2 → 5.

## 1. Analytic `call_and_ladj` for `BernsteinTransform`

`BernsteinTransform` inherits `MonotonicTransform.call_and_ladj`. That method
gets the Jacobian by `torch.autograd.grad` in the forward pass. It therefore
roughly doubles the training graph.

It also makes the per-node loss un-`torch.compile`-able. Autograd in the
forward pass gives a double backward, and `aot_autograd` does not support one.
Therefore the op-count axis stays closed in eager mode.

The derivative is closed form: `f'(x) = order·Δθ` against the order−1 basis.
zuko already computes it in `_offset_and_slope` for the tail slopes. An
override is therefore ~25 lines and needs no API change.
`MonotonicRQSTransform` already has such an override. tramdag gets a free
speedup and the `torch.compile` axis back.

## 2. Learnable/linear tails for `MonotonicRQSTransform`

zuko pads the knot derivatives with `exp(0)=1`. It extrapolates as the
identity outside `[-B, B]`. Therefore the tail slope stays fixed at 1 for
every θ. This is the structural reason why `spline` consistently trails
`bernstein` here, because ~10% of the data sits beyond the 5%/95% pre-scaling
range. CLAUDE.md and `spec.py` also describe this.

Upstream, zuko can accept boundary derivatives of shape `(*, K+1)`, or it can
take an opt-in `tails="linear"`. Identity tails are deliberate in the NSF
design. The framing must therefore stay back-compatible. This change will
delete the caveats and make `spline` a first-class transform choice.

## 3. Public inverse of `_constrain_theta`

`BernsteinUT.marginal_init_theta` inverts zuko's *private* `_constrain_theta`
(cumsum-of-softplus) in closed form. It hard-codes two internal details of
that function:

- the `log(2)·n/2` centering
- the tying of the first two and the last two diffs

Any upstream reparametrization therefore breaks the calibrated start silently.
A public `unconstrain_theta(theta)` that takes the target control points is
~15 lines upstream. It removes the coupling to these private details. The
framing is "identity/linear initialization support", a standard flow trick.

## 4. Docstring fix: the Bernstein θ-shape off-by-one

zuko's docstring says that θ has shape `(*, M-2)` for a degree-M polynomial.
`n` unconstrained coefficients become `n+2` control points, which is degree
`n+1`. The correct claim is therefore `(*, M-1)`. This off-by-one is a real
replication trap: order 21 against the paper's `len_theta=20`, which is order
19. It costs tramdag prose in three documents. The PR is trivial and an accept
is near-certain.

## 5. `Logistic` in `zuko.distributions`

Neither zuko nor torch ships a Logistic distribution. tramdag therefore
hand-rolls `StandardLogistic` in ~50 lines, with `log_prob`, `sample` and
`icdf`. Logistic latents are standard in transformation models and in discrete
flows, and zuko has a precedent in `GeneralizedNormal`. But zuko can point at
`TransformedUniform(SigmoidTransform().inv)` instead. tramdag keeps its
`_U_EPS` clamp locally in both cases. This candidate has the lowest value of
the five.

## Anti-candidates (look upstreamable, are not)

- **Quantile pre-scaling and `_ScaledUT`** — this is a modeling choice. zuko's
  `ComposedTransform` with `MonotonicAffineTransform` already composes it.
- **The ordinal ordered-logit transform and log-space `ordinal_log_prob`** —
  this transform is not a bijection. It stays outside zuko's flow scope. It is
  torch.distributions material, if anywhere.
- **LS, CS, CI and VC terms, the marginal-init policy, propensity centering
  and scores** — these are TRAM-DAG semantics. zuko's `MaskedMLP` already
  supports arbitrary adjacency. tramdag does not use it, because it needs
  per-term interpretability.
- **The spline `slope` knob and the Bernstein `eps` knob** — zuko already
  exposes both upstream.
