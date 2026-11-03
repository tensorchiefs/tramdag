# Notebook data

Small datasets the notebooks read. Anything larger, or tied to an experiment
rather than a notebook, belongs in `experiments/data/` instead.

## `birthwt.csv`

Five columns of `MASS::birthwt`, the low-birth-weight study of Hosmer and
Lemeshow. It holds 189 births at Baystate Medical Center.
`classical_fit_tram_dag.py` uses it as a logistic-regression example that a
reader can re-fit in R.

| column | meaning |
|---|---|
| `low` | birth weight below 2.5 kg (the outcome, 0/1) |
| `age` | mother's age in years |
| `lwt` | mother's weight at last menstrual period, lbs |
| `smoke` | smoking during pregnancy (0/1) |
| `bwt` | birth weight in grams, the continuous outcome that `low` is cut from |

Exported verbatim from MASS 7.3-58.2 under R 4.2.3:

```r
library(MASS)
write.csv(birthwt[, c("low", "age", "lwt", "smoke", "bwt")],
          "notebooks/data/birthwt.csv", row.names = FALSE)
```

The copy lets the notebook run without R. The R side reads `birthwt` from
MASS directly. `MASS` is GPL-2/GPL-3, and its datasets are redistributable
on those terms.

`classical_fit_tram_dag.py` compares against R coefficients from this file.
If you regenerate it, re-run the R snippet and update `R_GLM` and `R_LOGLIK`
there.

## `vaca.csv`

The bimodal VACA triangle of Section 1b of `classical_fit_tram_dag.py`: 1000 rows
of `x1`, `x2`, `x3` from the SCM stated in
[`docs/paper-replication.md`](../../docs/paper-replication.md).

The notebook holds its seeded generator in an `if False:` block, so a
normal run reads this file. Git tracks the file so that the R script fits
the same rows as the flow.

The notebook holds R's coefficients for this file in `R_COLR`, and
`classical_fit_tram_dag.R` produces them with `order = 21`
([`docs/zuko-upstream.md`](../../docs/zuko-upstream.md) on the count). If you
change `n` or the seed, re-run the R script and update `R_COLR`.

This file is not `experiments/data/vaca/`, the frozen benchmark under the
data contract of [`experiments/README.md`](../../experiments/README.md).
