# Paper replication: protocol, hyperparameters and results, per experiment

The eight variants under `experiments/` replicate the TRAM-DAG paper
(Sick & Dürr, CLeaR 2025, arXiv:2503.16206) against its own R code
(`tensorchiefs/tram-dag`). For each experiment this page lists the DGP, the
model, every hyperparameter with its source, what deviates from the paper and
why, and the numbers. The paper states four training numbers: n = 40000,
500 epochs, Adam, and Bernstein order 20; it shows its results as figures, so
where it gives no number the "paper" column names the figure and what it
shows. The pinned ground truth is `experiments/ground_truth/*.json`;
[experiments/README.md](../experiments/README.md) explains the YAML variants,
the frozen data and the check.

Seeds are DGP 42, init 7 and shuffle 0. The triangle scripts of the reference
run unseeded and the comparison scripts seed R's RNG, which torch cannot
replay, so every seed here is a repository choice.

## What is common to all eight

| item | reference | here |
|---|---|---|
| latent | standard logistic, shifts added on the continuous scale, subtracted for ordinal $P(Y \le k) = \sigma(\vartheta_k - s)$ | same ([notation.md](notation.md)) |
| Bernstein basis | `len_theta` unconstrained coefficients, `to_theta` softplus-cumsum, domain → [0, 1] with tangent-linear extrapolation outside; the domain comes from the train 5 %/95 % quantiles in the triangle scripts (`quantile(..., c(0.05, 0.95))`) and from min/max in the comparison scripts (`scale_df`) | zuko Bernstein, `n_coeffs` unconstrained (zuko ties two control points on, so `n_coeffs=20` is order 21 against the reference's order 19; the free-parameter count is what matches), domain = train `range_q`/1−`range_q` quantiles → [−5, 5], linear extrapolation. Triangle: `range_q: 0.05`, a match up to the reparametrization. CAREFL: `range_q: 0`, the reference's min/max. VACA: quantiles kept, deviation D1, measured below |
| networks | triangle scripts `create_param_net` with `hidden_features = c(2, 25, 25, 2)` continuous and `c(2, 2, 2, 2)` mixed, sigmoid (the ReLU line is commented out); the vector reads as in/out dims around the hidden stack, so the hidden layers are (25, 25) and (2, 2). Comparison scripts `make_model`: `dense(10, tanh) → dense(100, tanh) → dense(len_theta)`, one net per node | `units` and `activation` set per variant to exactly those stacks. The package defaults in [code-map.md](code-map.md) replicate the PyTorch reference `buehlpa/TramDag` instead and are not the paper's nets |
| init | triangle scripts: `LinearMasked` layers with Keras `random_normal` (N(0, 0.05²)) on weights and biases, the LS `beta` layer included; comparison scripts: `layer_dense` default, glorot-uniform weights and zero biases | `init: normal` (triangle) and `init: glorot` (VACA/CAREFL) through `CausalFlowDAG(init=)`; torch's default init remains the framework default, and under the full-batch protocol the init decides the fit (see VACA) |
| optimizer | Keras Adam, eps 1e-7 | torch Adam, eps 1e-8; measured: no effect (VACA identical to four digits) |
| calibrated start | none | `init_marginals: true` for the two triangle LS models, `atan-cs` and `exp-cs`, deviation D4 below; off where it moves the endpoint (`linear-cs`, `sin-cs`, VACA) and impossible for CAREFL's `range_q: 0` domain. `calibrate` never touches the weights |
| intercept output layer | Keras dense with bias | bias-free, deviation D3: the same function class, because the bias adds a constant to all unconstrained coefficients |
| plateau rule (VACA/CAREFL) | `update_learning_rate`: one optimizer, reduce when the summed validation NLL has not improved for 50 epochs (strict `<`), factor 0.1, min 1e-7 | torch `ReduceLROnPlateau(patience=49, threshold=0, threshold_mode="abs", factor=0.1, min_lr=1e-7)` on the summed `history["val"]`, the same rule, verified against torch's source; `experiments/helpers.py::fit_paper` drives it |

## Triangle, continuous (`triangle.py`): paper Sec. 6.1, App. C.3

**DGP** (`summerof24/triangle_structured_continous.R`):

- x1 ~ 0.5 N(0.25, 0.1²) + 0.5 N(0.73, 0.05²)
- h(x2 | x1) = 5 x2 + 2 x1 = u2
- h(x3 | x1, x2) = 0.63 x3 − 0.2 x1 − f(x2) = u3, u ~ logistic

f has three forms: `linear` −0.3 x, `atan` 0.75 atan(5 (x + 0.12)), `sin`
2 sin(3x) + x. True weights in the flow's convention: β12 = +2, β13 = −0.2,
β23 = +0.3 (linear only). A fitted `CS` learns −f(x2) + const.

**Model**: `x1: SI`, `x2: SI + LS(x1)`, `x3: SI + LS(x1) + {LS(x2) | CS(x2)}`,
Bernstein, the CS net hidden (25, 25) with sigmoid.

| hyperparameter | paper / R code | here |
|---|---|---|
| train / validation | `dgp(40000)` / `dgp(40000)`, two draws | 40000 / 40000, two draws |
| epochs | 500 | 300 (`linear-cs`: 500), one continuous run; the CI deviation |
| lr | 0.001 (`optimizer_adam()` default) | 0.004, the CI deviation |
| batch | 32 (Keras `fit()` default) | 256, the CI deviation |
| `len_theta` / `n_coeffs` | 20 | 20 |
| schedule / early stop / init | none / none, final weights / random_normal | none / none / `init: normal` |
| coefficient read-out | after every epoch (Keras loop) | `fit(callbacks=)`, every epoch |

**Results**

| variant | metric | paper | paper protocol, `init: normal` (batch 32 / lr 0.001 / 500 epochs) | CI config, the pinned ground truth |
|---|---|---|---|---|
| linear-ls | β12 / β13 / β23 | Fig. 14 trajectories; App. C.3 text: 1.98 / −0.21 / 0.26 | 1.987 / −0.170 / 0.282 | 1.981 / −0.161 / 0.281 |
| linear-cs | β13; max \|ĝ − (−f)\| on [−1, 1] | Fig. 17: fitted CS is a straight line | −0.173; 0.122 | −0.178; 0.088 |
| atan-cs | β13; cs max err | Fig. 7 right / 15 / 16: CS on −f; text: β12 = 2.07, β13 = −0.203 | −0.168; 0.059 | −0.168; 0.108 |
| sin-cs | β13; cs max err | Fig. 18: CS follows the non-monotone f on x2 ∈ [−1, 0] | −0.195; 1.10 | −0.168; 0.240 on the paper's [−1, 0] window |
| all | \|E[x3 \| do(x1 = −1)] flow − DGP\| | Figs. 16/17: histograms overlap | 0.09–0.14 | 0.08–0.11 |
| all | val NLL x3 | — | 2.4607–2.4764 | 2.4606–2.4764 |

β13 measures −0.16 to −0.18 instead of −0.2. β13 multiplies x1, whose two
mixture components sit at 0.25 and 0.73, so x1 has sd 0.254 against 0.375 for
x2 and 2.918 for x3, and SE(β13) ≈ 0.036 at n = 40000. The value stays within
about one SE, and β13 is too weakly identified at n = 5000 for the frozen
5000-row CSVs to serve as fit checks; the fit checks train on the paper's
n = 40000 protocol.

The (25, 25) hidden stack is what lands `sin-cs` on Fig. 18, including its
small −1-endpoint deviation; a literal `[2, 25, 25, 2]` reading puts a
2-sigmoid bottleneck on the input and cannot reproduce the figure at any
protocol (cs max err 1.22).

## Triangle, mixed (`triangle_mixed.py`): paper Sec. 6.2, App. C.4, App. B

**DGP** (`triangle_structured_mixed.R`): x1 and x2 as above. x3 is ordinal
with four levels and cutpoints θ = (−2, 0.42, 1.02) from the R code, which the
paper does not state. The level is #{k : u3 > θ_k + 0.2 x1 + f(x2)}, with f
either `linear` −0.3 x or `exp` 0.5 exp(x). The paper adds the ordinal shift
and the flow subtracts it, so the fitted weights are β13 = −0.2, β23 = +0.3.

**Model**: `x1: SI`, `x2: SI + LS(x1)`,
`x3: OrdinalNode(4, LS(x1) + {LS(x2) | CS(x2)})`, the CS net hidden (2, 2)
with sigmoid.

| hyperparameter | paper / R code | here |
|---|---|---|
| train / validation | `dgp(40000)` / `dgp(10000)` | 40000 / 10000, two draws |
| epochs, lr, batch, schedule, init | 500, 0.001, 32, none, random_normal | 350 (`exp-cs`) / 200 (`linear-ls`) continuous, none, `init: normal`; lr 0.004 (`linear-ls`: 0.002) / batch 256, the CI deviations |
| `n_coeffs` (x1, x2) | 20 | 20 |
| odds-ratio check (App. C.4) | odds(x2 ≤ −1) under do(x1 += 1), theory e² ≈ 7.39 | 40000 rows, seed 99 |
| counterfactual PMF (App. B) | Fig. 10 explains why point counterfactuals fail for the ordinal node | 2000 rows × 200 draws |

**Results**

| variant | metric | paper | paper protocol, `init: normal` (500 epochs) | CI config, pinned |
|---|---|---|---|---|
| linear-ls | β12 / β13 / β23 | Fig. 19: 2 / −0.2 / 0.3 | 1.987 / −0.246 / 0.319 | 1.978 / −0.258 / 0.333 |
| linear-ls | odds ratio predicted / DGP | e² ≈ 7.39; C.4 text: 7.74, CI [7.16, 8.38] | 7.29 / 7.19 | 7.23 / 7.19 |
| linear-ls | CF PMF TV vs analytic; P(true level) flow / analytic / mode bound | — (App. B qualitative) | 0.044; 0.718 / 0.728 / 0.806 | 0.050; 0.716 / 0.728 / 0.806 |
| exp-cs | β13; cs max err | Fig. 20: distributions match | −0.207; 0.142 | −0.200; 0.156 |
| exp-cs | CF PMF TV; P(true level) | — | 0.023; 0.917 / 0.921 | 0.024; 0.920 / 0.921 |

## VACA / CNF benchmark (`vaca.py`): paper Sec. 5.1–5.2, App. C.1

**DGP** (Sanchez-Martin et al. 2022, App. E.1): x1 ~ 0.5 N(−2, 1.5) + 0.5
N(1.5, 1), x2 = −x1 + N(0, 1), x3 = x1 + 0.25 x2 + N(0, 1). The noise is
Gaussian, outside the logistic-latent family, and the all-`CI` flow must fit
it. The analytic target is E[x3 | do(x2 = a)] = −0.25 + 0.25 a. The paper's
Sec. 5.2 text says a ∈ {−3, −2, 0}; its Fig. 5 panels and the R code
(`vaca_triangle.r`) intervene at a ∈ {−3, −1, 0}, and this repository follows
the code. do(x2 = −3) is off-manifold extrapolation and takes a looser
tolerance.

**Model**: `x1: SI`, `x2: CI(x1)`, `x3: CI(x1, x2)`, Bernstein
`n_coeffs = 31` (reference M = 30, `len_theta = 31`), the `make_model` nets.

| hyperparameter | R code | here |
|---|---|---|
| train / validation | nTrain 2500 / `dgp(5000)` | 2500 / 5000, two draws |
| epochs | 10000 (`Figure_Triangle_Linear_Bimodal.R`, the sourcing script, not in our copy of the R code, so EPOCHS/M/nTrain rest on that reading) | 10000, one run, 1:1 |
| lr | 0.001 | 0.001 |
| batch | full batch (one `apply_gradients` per epoch) | 2500 = n_train |
| schedule | the plateau rule above | the same rule; it fires at about epoch 9050 and freezes an all-bounds point |
| input scaling | `scale_df`: everything min-max to [0, 1] | `input_transform: minmax` on the CI terms |
| Bernstein domain | train min/max (`scale_df`) | 5 %/95 % quantiles (`range_q: 0.05`), deviation D1: at seed 7 the reference's min/max domain scores 0.289 / 0.040 / 0.067 against the quantiles' 0.096 / 0.080 / 0.022, worse at do(x2 = −3) and do(x2 = 0). CAREFL, same nets, measures the opposite way |
| n_compare | — | 50000 |

**Results**: the check is the flow's error against the analytic mean, not a
pinned flow value.

| metric | paper | here, the pinned ground truth |
|---|---|---|
| \|E[x3 \| do(x2 = −3)] − (−1.0)\| | Fig. 5: densities overlap | 0.096 |
| \|E[x3 \| do(x2 = −1)] − (−0.5)\| | Fig. 5 | 0.080 |
| \|E[x3 \| do(x2 = 0)] − (−0.25)\| | Fig. 5 | 0.022 |
| sd(x1) flow vs analytic 2.0767 | Fig. 4: bimodal x1 fitted (the default CNF fails) | 2.036, error 0.040 |
| val NLL x3 | — | 1.4427 |

The result is seed-sensitive at the off-manifold point do(x2 = −3), where the
error spans 0.03–0.27 over four init draws, against 0.005–0.026 at
do(x2 = 0). The paper shows one run's densities. The committed bound is 2.5×
the seed-7 measurement.

The table below traces what each protocol ingredient is worth under the
reference protocol, one change at a time, on the paper text's grid −3 / −2 / 0
with min-max inputs unless the row says otherwise.

| variant | error | reading |
|---|---|---|
| raw parents into the nets | 0.731 / 0.426 / 0.017 | the tanh nets saturate: 40 % of the rows have \|x1\| > 2, 43 % \|x2\| > 2 |
| global plateau, torch init | 0.523 / 0.334 / 0.129 | the summed NLL keeps improving through x1 while x3 overfits |
| … without any plateau | 0.562 / 0.354 / 0.142 | the rule is what stops the full-batch overfit |
| … Adam eps 1e-7 | 0.523 / 0.334 / 0.129 | no effect |
| … Bernstein on train min/max (D1 off) | 0.154 / 0.012 / 0.062 | helps, not the cause |
| … glorot init | 0.035 / 0.006 / 0.007 | the cause (a different random draw than the config's seed 7) |
| … glorot + min/max | 0.035 / 0.056 / 0.057 | |

A probe of the reference trajectory shows the do(x2) errors descend until
epoch 6000–8500 and then creep back up; the plateau anneal fires at about
9050 and freezes them. The reference protocol's quality is its anneal landing
inside that window. Rejected alternatives, each measured: relu with raw
parents wanders 0.17–0.45 on do(x2 = −3) and holds the bound only at isolated
epochs; tanh on raw parents saturates outright at 0.731; minibatch stepping
(batch 256, even with minmax + tanh) converges with a systematic −0.11
common-mode offset of all three do-means while the observational fit stays
perfect.

## CAREFL benchmark (`carefl.py`): paper Sec. 5.3, App. C.2

This benchmark is the reference run 1:1 on the reference's own data.
`carefl_fig5.r` sets `USE_EXTERNAL_DATA = TRUE` and trains on CAREFL's own
committed 2500 rows (`X.csv`, x3/x4 sd-standardized by 6.0104/1.9114) with
`val = train`. CAREFL's repository also commits the observation `xObs.csv`,
the analytic truth curves and its own predictions on the grid
`seq(-3, 2.9, 0.1)`. This repository freezes all of it under
`experiments/data/carefl-cf`, external input with no generator, so the
Fig. 6 curves are comparable point by point and every metric is in the
reference's standardized units.

**DGP** (Khemakhem et al. 2021): x1, x2 ~ Laplace(0, 1/√2),
x3 = x1 + 0.5 x2³ + ε, x4 = −x2 + 0.5 x1² + ε with ε ~ Laplace(0, 1/√2),
x3/x4 divided by their sample sds. Counterfactuals are analytic by noise
abduction. The observation is `xObs.csv` = (2, 1.5, 0.8465, −0.2616); the
paper prints (2, 1.5, 0.81, −0.28), slightly off it. The queries are
x3^cf | do(x2 = α) and x4^cf | do(x1 = α) on the committed grid.

**Model**: `x1, x2: SI`, `x3, x4: CI(x1, x2)`, Bernstein `n_coeffs = 31`,
the same `make_model` nets as VACA, `range_q: 0` for the reference's
`scale_df` min-max Bernstein domain.

| hyperparameter | R code (`carefl_fig5.r`) | here |
|---|---|---|
| train / validation | CAREFL's own `X.csv`, sd-standardized, `val = train` | the same committed `X.csv`, `val = train`; a fresh standardized draw is scored, never trained or annealed on |
| epochs, lr | 7000 @ 0.001 | 7000 @ 0.001 |
| batch, schedule, input scaling, init | full batch, plateau 0.1/50/1e-7, `scale_df`, glorot | same |
| Bernstein domain | train min/max (`scale_df`) | train min/max (`range_q: 0`) |
| scoring | the single `x_obs`, curves over α (Fig. 6) | the same, plus 300 held-out rows at α ∈ {−1.5, 0, 1.5}, all in standardized units |

**Results**, in standardized units.

| metric | paper / CAREFL | here, the pinned ground truth |
|---|---|---|
| Fig. 6 max \|x3^cf error\| | Fig. 6: flow tracks the DGP curve; CAREFL's committed x3 preds err up to ~0.7 at α = −3 | 0.066 |
| Fig. 6 max \|x4^cf error\| | CAREFL's committed x4 preds: max 0.174 | 0.204, at the α = −3 grid edge; 0.074 at α = 0 |
| held-out CF MAE x3, α = −1.5 / 0 / 1.5 | — | 0.015 / 0.009 / 0.013 |
| held-out CF MAE x4, α = −1.5 / 0 / 1.5 | — | 0.080 / 0.027 / 0.048 |

Three ingredients carry the agreement, each measured with everything else
held fixed. Training on the committed rows instead of a fresh 2500-row draw
cut the Fig. 6 x4 max from 0.40 to 0.34 in standardized units. The reference
optimization, 7000 @ 0.001 with `val = train`, fixed the parabola bottom
(error at α = 0 from 0.21 to 0.06). The min-max domain cut the grid edges
(x4 0.37 → 0.20, x3 0.20 → 0.07). On fresh draws the parabola bottom sits
below the truth near α = 0 for a reason that is neither protocol nor init:
finite-sample variance of the Laplace 12 % quantile at which the observed
noise sits, in the sparse region x1 = 2 with about 190 nearby rows; CAREFL's
own committed draw is a mild one, which is one more reason to train on it.

Rejected alternatives, each measured: raw parents saturate the sigmoid on the
Laplace parents as tanh does; relu underfits x3 (val NLL 1.46–1.47 against
the 1.403 ± 0.05 band) and grows a fragile Bernstein tail, with one held-out
row inverted to x4 ≈ 580; minibatch underweights the sparse x2 tail where the
Fig. 6 grid ends (cf MAE x3 at do(x2 = 1.5) 0.40 against the 0.319 bound, and
the Fig. 6 x3 error 5.9 against the full-batch 2.7).

## D4: the marginal start, measured per variant

`init_marginals` sets every simple intercept to its column's empirical
marginal before the first epoch (Bernstein: the control points follow
$\operatorname{logit}\hat F$; ordinal: the class log-odds). The reference
starts `bernp$beta` at zero. Measured on this machine at the CI protocol, one
run per variant with the start off and on, seeds unchanged:

| variant | epochs to within 0.01 of the final train NLL, off → on | endpoint, off → on | verdict |
|---|---|---|---|
| triangle linear-ls | 28 → 5 | identical (β to 4 digits, val NLL 2.4606) | on |
| triangle linear-cs | 28 → 5 | cs max err 0.025 → 0.056; do(x1) err 0.132 → 0.080 | off: the misspecified line is what this variant measures, and the start bends it |
| triangle atan-cs | 28 → 6 | cs max err 0.108 → 0.074; do(x1) err 0.095 → 0.072 | on |
| triangle sin-cs | 34 → 10 | cs max err 0.240 → 0.088; do(x1) err 0.075 → 0.201, past the 0.188 bound | off: a different optimum, better curve, worse L2 |
| mixed linear-ls | 42 → 9 | identical | on |
| mixed exp-cs | 28 → 6 | cs max err 0.143 → 0.060; do(x1) err 0.018 → 0.010 | on |
| VACA | within 0.5: 2245 → 168 | val NLL 1.4496 → 1.4348; do(x2 = 0) err 0.019 → 0.101, past the 0.044 bound | off: the plateau anneal fires elsewhere and freezes a different point |
| CAREFL | — | — | impossible: `range_q: 0` has no marginal start |

The start is a pure initialization, so where the optimizer reaches the same
basin the endpoint is identical and only the epoch count changes. Where the
model has a network shift or the anneal decides the endpoint, the basin
changes, and the four such cases split two to two. The pinned ground truth
of the four variants that switched on is unchanged: every metric stays
within its tolerance, and the check's advisory notes on their bounds are
listed for the next deliberate re-pin.

## Runtime and the CI deviations

The reference protocol's cost is the motivation for every deviation. The
triangle runs 500 epochs × 1250 steps of batch 32; one epoch takes 3.2 s
single-process at 2–4 torch threads and 5.5 s at 32 threads, because the
step is overhead-bound, so one variant takes 42–69 min on the 2-core CI
runners. The triangle scripts therefore use batch 256 at lr 0.004, 8× fewer
optimizer steps per epoch, chosen from this grid at the paper's 500 epochs
with glorot init. Every entry is the cs max err unless the column says
otherwise.

| batch / lr / epochs | linear-ls β13 | linear-cs | atan-cs | mixed exp-cs (TV) | steps vs paper |
|---|---|---|---|---|---|
| **32 / 0.001 / 500 (paper)** | −0.170 | 0.110 | 0.080 | 0.071 (0.019) | 1× |
| 128 / 0.001 / 500 | −0.170 | 0.170 | 0.041 | 0.126 (0.023) | 1/4 |
| 128 / 0.004 / 500 | −0.178 | 0.160 | 0.094 | 0.070 (0.014) | 1/4 |
| 128 / 0.004 / 250 | −0.170 | 0.147 | 0.070 | 0.131 (0.035) | 1/8 |
| **256 / 0.004 / 500 (CI)** | −0.171 | 0.132 | 0.073 | 0.072 (0.017) | 1/8 |
| 256 / 0.008 / 500 | −0.181 | 0.163 | 0.113 | 0.073 (0.017) | 1/8 |
| 512 / 0.010 / 500 | −0.169 | 0.183 | 0.104 | 0.119 (0.015) | 1/16 |

Batch 256 / lr 0.004 is the only row that keeps every cs error within about
0.02 of the paper protocol. Larger batches or rates bend the misspecified
linear-cs curve (0.16–0.18), and lr 0.008 hurts atan-cs. At 500 epochs a
triangle job takes 7–11 min instead of 42–69.

The epoch floors, measured at that batch and rate:

- **Triangle** holds every bound at 300 epochs; at 150 the run fails atan's
  do(x1) bound (0.235 > 0.206). `linear-cs` keeps 500, because at 300 its
  fitted cs curve flattens past |x2| > 0.5 (max err 0.140 against 0.088): the
  DGP line spans only ±0.3, and the sigmoid net's slow tail convergence is half
  of that. relu is out, because the 2-unit input layer dies with it (atan:
  β13 +0.53, cs err 1.42).
- **Triangle-mixed** `exp-cs` has two slow parts, the 2-unit sigmoid CS net
  and the do(x1) mean: at 100 / 250 / 350 epochs the cs max err is
  0.344 / 0.177 / 0.156 and the do err 0.033 / 0.060 / 0.022, so it trains
  350. `linear-ls` has no CS net to wait for and trains 200 epochs at
  lr 0.002, which reads the weakly identified β13/β23 slightly closer to the
  500-epoch values than lr 0.004 does. relu dies at the sd-0.05 normal init
  (cs err 0.859, β13 −0.055).
- **VACA and CAREFL** run their references 1:1; the rejected shortcuts are in
  their sections.

## Repository choices the paper does not state

- the seeds,
- the validation draw sizes for the triangle scripts (the R code's `test`),
- `n_compare`,
- `n_heldout = 300` and the α set {−1.5, 0, 1.5}, scored next to the single
  `x_obs`,
- the mixed cutpoints from the R code,
- the odds-ratio sample (40000 rows),
- the counterfactual-PMF sample (2000 rows × 200 draws).
