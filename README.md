# Graph dimension of a chaotic flow — code and data

Code and numerical data behind the Letter *"The graph dimension of a chaotic flow is not in its
covariance spectrum."*

## Layout

```
code/analysis/                 computation scripts (the v-series, in the order they were written)
code/analysis/verification/    independent re-implementations used to audit the analysis
code/figures/make_figures.py   the only figure script; reads data/*.txt and writes the three figures
data/*.txt                     every number plotted in the Letter, tab-separated with '#' headers
```

## Reproducing the figures

Every figure in the Letter is regenerated **from the tabulated data alone**, with no simulation:

```
python code/figures/make_figures.py
```

It reads `data/*.txt` and writes `PRL_Fig1.{png,pdf}`, `PRL_Fig2.{png,pdf}` and
`PRL_Fig3.{png,pdf}`. Each figure has four panels; the panel order matches the captions in the
Letter.

## Reproducing the numbers

The analysis scripts are standalone and deterministic given the snapshot sets. Entry points for the
results quoted in the Letter:

| result | script |
|---|---|
| the seven Kuramoto–Sivashinsky elbows | `v20_robust.py` |
| graph slope and the criterion margin | `v50_graph_slope.py`, `v95_multi_prior.py` |
| Kaplan–Yorke dimensions and `D_KY/N` | `v203_negatives_dim.py` |
| the criterion across every system and prior | `v100_ten_checks.py` |
| the power window | `v200_negative_power.py`, `v201_l66_bootstrap.py` |
| the complex Ginzburg–Landau exponents | `v206_cgle_close.py` |
| the parity law, exactly-`k`-degree construction | `recalc_parity.py` |

## Data files

| file | figure or table |
|---|---|
| `fig1_covariance_blindness.txt` | Fig. 1 |
| `fig2a_fourth_moment_routes.txt` | Fig. 2(a) |
| `fig2b_fourth_moment_analytic.txt` | Fig. 2(b) |
| `fig2c_parity_grid.txt` | Fig. 2(c) — 16 combinations, exactly-`k` construction |
| `fig2a_elbow_significance.txt` | Fig. 2(d) |
| `fig3a_matched_accuracy.txt` | Fig. 3(a) |
| `fig3b_criterion_all.txt` | Fig. 3(b) — all fourteen systems |
| `fig2b_dimension_vs_domain.txt` | Fig. 3(c) |
| `fig2c_mackey_glass.txt`, `mackey_glass_robustness.txt` | Fig. 3(d) |
| `lyapunov_spectra.txt` | Kaplan–Yorke dimensions and `D_KY/N` |
| `fourth_moment_parity.txt`, `verification_fourth_moment.txt` | Supplemental S2 |
| `criterion_all_priors.txt` | Supplemental S4, prior dependence |

## A note on the numbers

`v200_negative_power.py` uses a construction that turned out to be ill-posed: it injects a graph in
POD coordinates and then tests with a Fourier prior, so the two are not aligned and the margin is
forced negative. It is retained because the correction is instructive and because
`v201_l66_bootstrap.py` and `v203_negatives_dim.py` supersede its conclusion. The superseding route
is the one quoted in the Letter.
