# Notebook data

Small datasets the notebooks read. Anything larger, or tied to an experiment
rather than a notebook, belongs in `experiments/<area>/data/` instead.

## `birthwt.csv`

Five columns of `MASS::birthwt`, the low-birth-weight study of Hosmer and
Lemeshow (1989). It holds 189 births at Baystate Medical Center.
`classical_fit_tram_dag.py` uses it as a logistic-regression example that a
reader can re-fit in R.

| column | meaning |
|---|---|
| `low` | birth weight below 2.5 kg (the outcome, 0/1) |
| `age` | mother's age in years |
| `lwt` | mother's weight at last menstrual period, lbs |
| `smoke` | smoking during pregnancy (0/1) |
| `bwt` | birth weight in grams — the continuous outcome `low` is cut from |

Exported verbatim from MASS 7.3-58.2 under R 4.2.3:

```r
library(MASS)
write.csv(birthwt[, c("low", "age", "lwt", "smoke", "bwt")],
          "notebooks/data/birthwt.csv", row.names = FALSE)
```

The copy exists only so the notebook runs without R. **The R side of the
comparison needs no file at all.** `birthwt` ships with MASS, so the notebook's
pasteable snippet reads `data = birthwt` directly. Anyone with an R install can
therefore check the three-way agreement. `MASS` is GPL-2/GPL-3, and its
datasets are redistributable on those terms.

Regenerating this file is a **contract change**. `classical_fit_tram_dag.py`
compares against R coefficients hard-coded from the fit above. A regenerated
CSV therefore means you re-run the R snippet and update `R_GLM` and
`R_LOGLIK` there.

## `vaca.csv`

The bimodal VACA triangle of Section 1 of `classical_fit_tram_dag.py`: 1000 rows
of `x1`, `x2`, `x3` from

```
x1 ~ 0.5 N(-2, 1.5) + 0.5 N(1.5, 1)
x2 = -x1 + N(0, 1)
x3 =  x1 + 0.25 x2 + N(0, 1)
```

The notebook carries its generator in an `if False:` block, so a normal run
reads this file and does not rewrite it. Run that block by hand to regenerate
the sample, which is seeded and therefore reproducible. The file is tracked so
that the R snippet in the same section reads the identical rows the flow was
fitted on.

The notebook pins R's reference coefficients for this file in its `R_COLR`
constant, and `classical_fit_tram_dag.R` is the script that produces them.
Use `order = 21`, not 19: the notebook's `n_coeffs = 20` counts
unconstrained coefficients, and `tram` counts one more. The notebook
explains that difference where it makes the comparison.

A change to `n` or to the seed changes those numbers. Re-run
`Rscript notebooks/classical_fit_tram_dag.R` and update `R_COLR` when you
change either.

Not to be confused with `experiments/paper/data/vaca/` — that is the frozen
5000-row benchmark of the TRAM-DAG paper replications, under the testing
contract described in CLAUDE.md. This file is neither frozen nor a contract.
