# Paper replication: protocol, hyperparameters and results, per experiment

The eight variants under `experiments/` replicate the TRAM-DAG paper
[@sick2025tramdag] against its own R code
(`tensorchiefs/tram-dag`). For each experiment this page lists the DGP, the
model, every hyperparameter with its source, what deviates from the paper and
why, and the numbers. The paper states five training numbers: n = 40000,
500 epochs, Adam, lr 0.001 and Bernstein `len_theta` 20 (order 19). It shows
its results as figures. Where it gives no number, the "paper" column names the
figure and what it shows. `experiments/ground_truth/*.json` holds the bounds and
centers. The tables show the measurements of the pinned run.
[experiments/README.md](../experiments/README.md) explains the YAML variants,
the frozen data and the check. The tables of D4, D5 and of the runtime section
are study runs of their own. They are not the pinned run.

Seeds are DGP 42, init 7 and shuffle 0. The triangle scripts of the reference
run unseeded. The comparison scripts seed R's RNG, which torch cannot
replay. So every seed here is a repository choice.

## What is common to all eight

| item | reference | here |
|---|---|---|
| latent | standard logistic, shifts added on the continuous scale, subtracted for ordinal $P(Y \le k) = \sigma(\vartheta_k - s)$ | same ([notation.md](notation.md)) |
| Bernstein basis | `len_theta` unconstrained coefficients, `to_theta` softplus-cumsum, domain → [0, 1] with tangent-linear extrapolation outside; the domain comes from the train 5 %/95 % quantiles in the triangle scripts (`quantile(..., c(0.05, 0.95))`) and from min/max in the comparison scripts (`scale_df`) | zuko Bernstein, `n_coeffs` unconstrained (zuko ties two control points on, so `n_coeffs=20` is order 21 against the reference's order 19; the free-parameter count is what matches), domain = train `range_q`/1−`range_q` quantiles → [−5, 5], linear extrapolation. Triangle: `range_q: 0.05`, a match up to the reparametrization. VACA and CAREFL: `range_q: 0`, the reference's min/max |
| networks | triangle scripts `create_param_net` with `hidden_features = c(2, 25, 25, 2)` continuous and `c(2, 2, 2, 2)` mixed, sigmoid (the ReLU line is commented out); the vector reads as in/out dims around the hidden stack, so the hidden layers are (25, 25) and (2, 2). Comparison scripts `make_model`: `dense(10, tanh) → dense(100, tanh) → dense(len_theta)`, one net per node | `units` and `activation` set per variant to exactly those stacks. The package defaults in [code-map.md](code-map.md) replicate the PyTorch reference `buehlpa/TramDag` instead and are not the paper's nets |
| init | triangle scripts: `LinearMasked` layers with Keras `random_normal` (N(0, 0.05²)) on weights and biases, the LS `beta` layer included; comparison scripts: `layer_dense` default, glorot-uniform weights and zero biases | `init: normal` (triangle) and `init: glorot` (VACA/CAREFL) through `CausalFlowDAG(init=)`; torch's default init is the framework default, and under the full-batch protocol the init decides the fit (see VACA) |
| optimizer | Keras Adam, eps 1e-7 | torch Adam, eps 1e-8; measured: no effect (VACA identical to four digits) |
| calibrated start | none | `marginal_init: true` for the two triangle LS models, `atan-cs` and `exp-cs`, deviation D4 below; off where it moves the endpoint (`linear-cs`, `sin-cs`) and impossible for the `range_q: 0` domain of VACA and CAREFL. Calibration without the flag never touches the weights |
| intercept output layer | Keras dense with bias | bias-free, deviation D3: the same function class, because the bias adds a constant to all unconstrained coefficients |
| plateau rule (VACA/CAREFL) | `update_learning_rate`: one optimizer, reduce when the summed validation NLL has not improved for 50 epochs (strict `<`), factor 0.1, min 1e-7 | torch `ReduceLROnPlateau(patience=49, threshold=0, threshold_mode="abs", factor=0.1, min_lr=1e-7)`, the same rule verified against torch's source, but per node on the node's own validation NLL, deviation D5 below; `experiments/helpers.py::fit_paper` drives it |

The deviations have these labels:

- **D3** is the bias-free intercept output layer (the table above).
- **D4** is the marginal start (the section D4 below).
- **D5** is the plateau rule per node (the section D5 below).

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

The scripts follow the R code in these points. Each epoch shuffles one
compiled model, so one continuous optimizer run fits it. No term has an
`input_transform`, so the nets read raw parents. The Bernstein domain comes from
the train 5 %/95 % quantiles. The fit keeps the final weights. The `linear-cs`
variant is the misspecified case of App. C.3.3 (Fig. 17): the DGP effect is
linear, and the complex shift must recover the straight line instead of bending
to follow the sample. `atan-cs` is the headline complex-shift run (Fig. 7, 15,
16). `sin-cs` shows that the recovery is not specific to atan, and its grid
covers x2 ∈ [−1, 0], the half period that Fig. 18 plots.

**Results**

| variant | metric | paper | paper protocol, `init: normal` (batch 32 / lr 0.001 / 500 epochs) | CI config, pinned run (measured) |
|---|---|---|---|---|
| linear-ls | β12 / β13 / β23 | Fig. 14 trajectories; Sec. 6.1 text: 1.98 / −0.21 / 0.26 | 1.987 / −0.170 / 0.282 | 1.986 / −0.153 / 0.261 |
| linear-cs | β13; max \|ĝ − (−f)\| on [−1, 1] | Fig. 17: fitted CS is a straight line | −0.173; 0.122 | −0.176; 0.031 |
| atan-cs | β13; cs max err | Fig. 7 right / 15 / 16: CS on −f; text: β12 = 2.07, β13 = −0.203 | −0.168; 0.059 | −0.156; 0.053 |
| sin-cs | β13; cs max err | Fig. 18: CS follows the non-monotone f on x2 ∈ [−1, 0] | −0.195; 1.10 | −0.165; 0.255 on the paper's [−1, 0] window |
| all | \|E[x3 \| do(x1 = −1)] flow − DGP\| | Figs. 16/17: histograms overlap | 0.09 to 0.14 | 0.031 to 0.125 (linear-cs, atan-cs, sin-cs); linear-ls 0.115 |
| all | val NLL x3 | (none) | 2.4607 to 2.4764 | 2.4608 to 2.4632 |

The pinned column gives the measurement of the pinned run, from its
`metrics.json`. A `value` entry of a run metric in
`experiments/ground_truth/*.json` is this measurement. The DGP, analytic and
CAREFL entries are constants. Each `max` bound on an error measure sits at 1.5x
to 4x of it. The `fit_seconds` bounds are 3x the time on the CI runner. No cell
in this column is a bound.

β13 measures −0.15 to −0.18 instead of −0.2 in the continuous triangle variants.
β13 multiplies x1, whose two mixture components sit at 0.25 and 0.73. So x1 has
sd 0.254 against 0.375 for x2 and 2.918 to 3.748 for x3. At n = 40000, SE(β13)
is about 0.036. The value is within about 1.3 SE of the truth. At n = 5000,
β13 is too weakly identified for the frozen 5000-row CSVs to serve as fit
checks. So the fit checks train on the paper's n = 40000 protocol.

The (25, 25) hidden stack is what lands `sin-cs` on Fig. 18, including its
small −1-endpoint deviation. A literal `[2, 25, 25, 2]` reading puts a
2-sigmoid bottleneck on the input. It cannot reproduce the figure at any
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

The scripts follow the R code as in the continuous triangle. They use one
compiled model with shuffled batches and keep the final weights. They have no
schedule and no `input_transform`, and the domain is the 5 %/95 % quantiles. The
CS net uses hidden (2, 2) with sigmoid, read as in/out dims around the hidden
stack like the continuous (25, 25). `exp-cs` pairs the ordinal node with a
nonlinear f. `linear-ls` is the headline mixed case (Fig. 9, 19, and the C.4
odds-ratio check). It is the only mixed variant with a true β23.

**Results**

| variant | metric | paper | paper protocol, `init: normal` (500 epochs) | CI config, pinned |
|---|---|---|---|---|
| linear-ls | β12 / β13 / β23 | Fig. 19: 2 / 0.2 / −0.3 (paper convention; −0.2 / +0.3 in the flow's) | 1.987 / −0.246 / 0.319 | 1.979 / −0.258 / 0.318 |
| linear-ls | odds ratio predicted / DGP | e² ≈ 7.39; C.4 text: 7.74, CI [7.16, 8.38] | 7.29 / 7.19 | 7.23 / 7.19 |
| linear-ls | CF PMF TV vs analytic; P(true level) flow / analytic / mode bound | (App. B qualitative) | 0.044; 0.718 / 0.728 / 0.806 | TV 0.050; 0.715 / 0.728 / 0.806 |
| exp-cs | β13; cs max err | Fig. 20: distributions match | −0.207; 0.142 | −0.209; 0.064 |
| exp-cs | CF PMF TV; P(true level) | (none) | 0.023; 0.917 / 0.921 | 0.020; 0.920 / 0.921 |

The pinned column follows the rule of the continuous table: every cell is a
measurement.

## VACA / CNF benchmark (`vaca.py`): paper Sec. 5.1 and 5.2, App. C.1

**DGP** [@sanchezmartin2022vaca, App. E.1]: x1 ~ 0.5 N(−2, 1.5) + 0.5
N(1.5, 1). x2 = −x1 + N(0, 1). x3 = x1 + 0.25 x2 + N(0, 1). The error term is
Gaussian, outside the logistic-latent family, and the all-`CI` flow must fit
it. The analytic target is E[x3 | do(x2 = a)] = −0.25 + 0.25 a. The paper's
Sec. 5.2 text says a ∈ {−3, −2, 0}. Its Fig. 5 panels and the R code
(`Figure_Triangle_Linear_Bimodal.R`) intervene at a ∈ {−3, −1, 0}, and this
repository follows the code. do(x2 = −3) is off-manifold extrapolation and takes
a looser tolerance.

**Model**: `x1: SI`, `x2: CI(x1)`, `x3: CI(x1, x2)`, Bernstein
`n_coeffs = 31` (reference M = 30, `len_theta = 31`), the `make_model` nets.

| hyperparameter | R code | here |
|---|---|---|
| train / validation | nTrain 2500 / `dgp(5000)` | 2500 / 5000, two draws |
| epochs | 10000 (`comparison/Figure_Triangle_Linear_Bimodal.R`: EPOCHS = 10000, M = 30, nTrain = 2500) | 10000, one run, 1:1 |
| lr | 0.001 | 0.001 |
| batch | full batch (one `apply_gradients` per epoch) | 2500 = n_train |
| schedule | the plateau rule above | the same rule per node, deviation D5 |
| input scaling | `scale_df`: everything min-max to [0, 1] | `input_transform: minmax` on the CI terms |
| Bernstein domain | train min/max (`scale_df`) | train min/max (`range_q: 0`), as in CAREFL |
| n_compare | (none) | 50000 |

**Results**: the check is the flow's error against the analytic mean, not a
pinned flow value.

| metric | paper | here, pinned run (measured) |
|---|---|---|
| \|E[x3 \| do(x2 = −3)] − (−1.0)\| | Fig. 5: densities overlap | 0.074 |
| \|E[x3 \| do(x2 = −1)] − (−0.5)\| | Fig. 5 | 0.001 |
| \|E[x3 \| do(x2 = 0)] − (−0.25)\| | Fig. 5 | 0.052 |
| sd(x1) flow vs analytic 2.0767 | Fig. 4: bimodal x1 fitted; the default CNF [@javaloy2023causalflows] fails | 2.064, error 0.012 |
| val NLL x3 | (none) | 1.4335 |

The do(x2) errors vary with the init seed. D5 below gives five seeds. At
do(x2 = −1) the per-node error goes from 0.001 to 0.018, and the config's
seed 7 gives the smallest. The paper shows one run's densities. The
committed bounds sit at 1.5x to 4x of the pinned measurement.

The table below traces what each protocol ingredient is worth under the
reference protocol with its global plateau, one change at a time. It uses the
paper text's grid −3 / −2 / 0 and min-max inputs unless the row says
otherwise.

| variant | error | reading |
|---|---|---|
| raw parents into the nets | 0.731 / 0.426 / 0.017 | the tanh nets saturate: 40 % of the rows have \|x1\| > 2, 43 % \|x2\| > 2 |
| global plateau, torch init | 0.523 / 0.334 / 0.129 | the summed NLL keeps improving through x1 while x3 overfits |
| … without any plateau | 0.562 / 0.354 / 0.142 | the rule is what stops the full-batch overfit |
| … Adam eps 1e-7 | 0.523 / 0.334 / 0.129 | no effect |
| … Bernstein on train min/max | 0.154 / 0.012 / 0.062 | helps, not the cause |
| … glorot init | 0.035 / 0.006 / 0.007 | the cause (a different random draw than the config's seed 7) |
| … glorot + min/max | 0.035 / 0.056 / 0.057 | |

Rejected alternatives:

- relu with raw parents holds the bound only at isolated epochs.
- tanh on raw parents saturates (first table row).
- Minibatch stepping leaves a common-mode offset in all three do-means,
  although the observational fit is perfect.

## CAREFL benchmark (`carefl.py`): paper Sec. 5.3, App. C.2

This benchmark is the reference run 1:1 on the reference's own data, except
D5 (the plateau per node).
`carefl_fig5.r` sets `USE_EXTERNAL_DATA = TRUE` and trains on CAREFL's own
committed 2500 rows (`X.csv`, x3/x4 sd-standardized by 6.0104/1.9114) with
`val = train`. CAREFL's repository also commits the observation `xObs.csv`,
the analytic truth curves and its own predictions on the grid
`seq(-3, 2.9, 0.1)`. This repository freezes all of it under
`experiments/data/carefl-cf`, external input with no generator, so the
Fig. 6 curves are comparable point by point and every metric is in the
reference's standardized units.

**DGP** [@khemakhem2021carefl]: x1, x2 ~ Laplace(0, 1/√2).
x3 = x1 + 0.5 x2³ + ε, x4 = −x2 + 0.5 x1² + ε with ε ~ Laplace(0, 1/√2),
x3/x4 divided by their sample sds. Counterfactuals are analytic by abduction
of the error term. The observation is `xObs.csv` = (2, 1.5, 0.8465, −0.2616);
the paper prints (2, 1.5, 0.81, −0.28), slightly off it. The queries are x3^cf |
do(x2 = α) and x4^cf | do(x1 = α) on the committed grid.

**Model**: `x1, x2: SI`, `x3, x4: CI(x1, x2)`, Bernstein `n_coeffs = 31`,
the same `make_model` nets as VACA, `range_q: 0` for the reference's
`scale_df` min-max Bernstein domain.

| hyperparameter | R code (`carefl_fig5.r`) | here |
|---|---|---|
| train / validation | CAREFL's own `X.csv`, sd-standardized, `val = train` | the same committed `X.csv`, `val = train`; a fresh standardized draw is scored, never trained or annealed on |
| epochs, lr | 7000 @ 0.001 | 7000 @ 0.001 |
| batch, schedule, input scaling, init | full batch, plateau 0.1/50/1e-7, `scale_df`, glorot | same, the plateau per node (D5) |
| Bernstein domain | train min/max (`scale_df`) | train min/max (`range_q: 0`) |
| scoring | the single `x_obs`, curves over α (Fig. 6) | the same, plus 300 held-out rows at α ∈ {−1.5, 0, 1.5}, all in standardized units |

**Results**, in standardized units.

| metric | paper / CAREFL | here, pinned run (measured) |
|---|---|---|
| Fig. 6 max \|x3^cf error\| | Fig. 6: flow tracks the DGP curve; CAREFL's committed x3 preds err up to 1.21 at α = −3 | 0.050 |
| Fig. 6 max \|x4^cf error\| | CAREFL's committed x4 preds: max 0.174 | 0.157, at α = 0; 0.149 at the α = −3 grid edge |
| held-out CF MAE x3, α = −1.5 / 0 / 1.5 | (none) | 0.014 / 0.009 / 0.013 |
| held-out CF MAE x4, α = −1.5 / 0 / 1.5 | (none) | 0.059 / 0.025 / 0.052 |

Two ingredients carry the agreement. One study run per ingredient changes
only that ingredient. Training on a fresh 2500-row draw instead of the
committed rows raises the Fig. 6 error. It is 0.21 for x4 and 0.11 for x3.
The 5 %/95 % domain instead of min-max raises it too, to 0.27 for x4 and 0.29
for x3. Both maxima sit at the α = −3 grid edge, where min-max gives 0.15 and
0.04. The reference protocol gives 0.16 for x4 and 0.05 for x3.

The largest x4 error of the reference run sits at α = 0, the bottom of the
parabola, where the flow lies above the truth. The fresh-draw run lies above
the truth at α = 0 too, by 0.12 for x4.

Rejected alternatives:

- Raw parents saturate the sigmoid on the Laplace parents, as tanh does.
- relu underfits x3 and grows a fragile Bernstein tail. One held-out row
  inverts to x4 ≈ 580.
- Minibatch underweights the sparse x2 tail where the Fig. 6 grid ends. The
  held-out x3 error at do(x2 = 1.5) and the Fig. 6 x3 error both grow.

## D4: the marginal start, measured per variant

`marginal_init` starts every simple intercept at its column's empirical
marginal ([fitting.md](fitting.md#the-marginal-start)). The reference starts
`bernp$beta` at zero. Measured at the CI protocol, one run per variant with
the start off and on, seeds unchanged:

| variant | epochs to within 0.01 of the final train NLL, off → on | endpoint, off → on | verdict |
|---|---|---|---|
| triangle linear-ls | 28 → 5 | identical (β to 4 digits, val NLL 2.4606) | on |
| triangle linear-cs | 28 → 5 | cs max err 0.025 → 0.056; do(x1) err 0.132 → 0.080 | off: the misspecified line is what this variant measures, and the start bends it |
| triangle atan-cs | 28 → 6 | cs max err 0.108 → 0.074; do(x1) err 0.095 → 0.072 | on |
| triangle sin-cs | 34 → 10 | cs max err 0.240 → 0.088; do(x1) err 0.075 → 0.201, past the 0.188 bound | off: a different optimum, better curve, worse L2 |
| mixed linear-ls | 42 → 9 | identical | on |
| mixed exp-cs | 28 → 6 | cs max err 0.143 → 0.060; do(x1) err 0.018 → 0.010 | on |
| VACA, CAREFL | (none) | (none) | impossible: `range_q: 0` has no marginal start |

The start is only an initialization. Where the optimizer reaches the same
basin, the endpoint is identical. Only the epoch count changes. Where the
model has a network shift or the anneal decides the endpoint, the basin
changes. Of the four such cases with a marginal start, two switch it on and
two leave it off.

## D5: the plateau rule per node

The reference reduces the learning rate of one optimizer when the summed
validation NLL stops improving. The replication fits every node on its own
with `fit`, so no summed NLL exists during its fit. Each node runs the same
rule on its own validation NLL, with its own optimizer. The rule and its
constants are those of the reference.

The table gives the VACA do(x2) errors at a = −3 / −1 / 0. It also gives the
x3 validation NLL, with one run per init seed on the same data. The global
column runs the reference rule through `fit_jointly`: one plateau on the
summed validation NLL.

| init seed | per node (here) | global (reference) |
|---|---|---|
| 1 | 0.087 / 0.006 / 0.039, 1.4339 | 0.046 / 0.111 / 0.093, 1.4384 |
| 2 | 0.077 / 0.008 / 0.042, 1.4339 | 0.357 / 0.212 / 0.123, 1.4379 |
| 3 | 0.103 / 0.018 / 0.036, 1.4341 | 0.036 / 0.088 / 0.056, 1.4385 |
| 4 | 0.096 / 0.011 / 0.040, 1.4337 | 0.011 / 0.062 / 0.047, 1.4367 |
| 7 (config) | 0.074 / 0.001 / 0.052, 1.4335 | 0.095 / 0.093 / 0.054, 1.4383 |

The per-node rule gives a lower x3 validation NLL on every seed. Its error at
do(x2 = −1) and do(x2 = 0) is smaller on every seed. At the off-manifold
point do(x2 = −3) the global rule errs less on three of the five seeds.

## Runtime and the CI deviations

The reference protocol is too slow for the 2-core CI runners. The triangle runs
500 epochs × 1250 steps of batch 32, and the step is overhead-bound. The
triangle scripts therefore use batch 256 at lr 0.004, with 8× fewer optimizer
steps per epoch. The grid below, at the paper's 500 epochs with glorot init,
selects them. Every entry is the cs max err unless the column says otherwise.

| batch / lr / epochs | linear-ls β13 | linear-cs | atan-cs | mixed exp-cs (TV) | steps vs paper |
|---|---|---|---|---|---|
| **32 / 0.001 / 500 (paper)** | −0.170 | 0.110 | 0.080 | 0.071 (0.019) | 1× |
| 128 / 0.001 / 500 | −0.170 | 0.170 | 0.041 | 0.126 (0.023) | 1/4 |
| 128 / 0.004 / 500 | −0.178 | 0.160 | 0.094 | 0.070 (0.014) | 1/4 |
| 128 / 0.004 / 250 | −0.170 | 0.147 | 0.070 | 0.131 (0.035) | 1/8 |
| **256 / 0.004 / 500 (CI batch and lr)** | −0.171 | 0.132 | 0.073 | 0.072 (0.017) | 1/8 |
| 256 / 0.008 / 500 | −0.181 | 0.163 | 0.113 | 0.073 (0.017) | 1/8 |
| 512 / 0.010 / 500 | −0.169 | 0.183 | 0.104 | 0.119 (0.015) | 1/16 |

Batch 256 / lr 0.004 is the only row that keeps every cs error within about
0.02 of the paper protocol. Larger batches or rates bend the misspecified
linear-cs curve (0.16 to 0.18), and lr 0.008 hurts atan-cs.

The epoch floors, measured at that batch and rate:

- **Triangle** holds every bound at 300 epochs. At 150 the run fails atan's
  do(x1) bound (0.235 > 0.200). `linear-cs` trains 500, because at 300 its fitted
  cs curve flattens past |x2| > 0.5. The DGP line spans only ±0.3. relu is out: it
  fits atan badly.
- **Triangle-mixed** `exp-cs` has two slow parts, the 2-unit sigmoid CS net and
  the do(x1) mean, so it trains 350 epochs. `linear-ls` has no CS net to wait
  for, and at lr 0.002 it trains 200 epochs. The lower lr 0.002 lands closer to
  the 500-epoch values. relu at the sd-0.05 normal init fails here too.
- **VACA and CAREFL** run their references 1:1 except D5;
  the rejected shortcuts are in their sections.

## Repository choices the paper does not state

- the seeds (the reference seeds R's RNG with 42, which torch cannot
  replay; a full-batch run ignores the shuffle seed),
- the validation draw sizes for the triangle scripts (the R code's `test`),
- `n_compare`, and `n_scatter` = 2000 points per VACA scatter panel (Fig. 4),
- `n_heldout = 300` and the α set {−1.5, 0, 1.5}, scored next to the single
  `x_obs`, drawn at the seed offset 999 from `dgp_seed`,
- the mixed cutpoints from the R code,
- the odds-ratio sample (40000 rows),
- the counterfactual-PMF sample (2000 rows × 200 draws).
