# Objective-Dependent Active Learning and Calibrated Uncertainty for logD Prediction

A reproducible benchmark of pool-based active learning (AL) and uncertainty
quantification for predicting distribution coefficients (logD) in rare-earth /
f-element solvent extraction, with prospective validation.

Author: Krishna Harish, Lawrence E. Elkins High School, Missouri City, Texas, USA.
Manuscript prepared for ACS Omega (see `paper/` and `submission/`).

## What this study finds

On real experimental data, with multi-seed statistics:

1. **A simple baseline beats the published deep network.** A random forest on the
   published features reaches R2 = 0.94 / RMSE = 0.33 on the validation split,
   versus the reported deep network (R2 = 0.85).
2. **The optimal AL acquisition is objective-dependent.** Exploitation (greedy, UCB)
   recovers all top extractants (top-10 recall = 1.0) but yields a poor predictive
   model (R2 = 0.63); exploration/random gives the best model (R2 ~ 0.85) but finds
   few top extractants. No strategy wins both; expected improvement and a hybrid
   portfolio trace the Pareto front. The same dichotomy reproduces on an independent
   4,200-compound logD benchmark (MoleculeNet Lipophilicity), establishing it as a
   general property of pool-based AL.
3. **AL does not help prediction.** Random selection matches or beats uncertainty
   sampling for predictive R2 at every budget.
4. **Calibration matters for trust, not acquisition.** Split-conformal calibration
   cuts the calibration error eight-fold (to within 1.3% of nominal coverage), but
   leaves AL acquisition unchanged, since acquisition depends only on the ranking of
   uncertainty.

Prospective validation on four independently synthesised ligands gives R2 = 0.69.

## Data

- `data/lipophilicity.csv`, `data/lipo_features.npz` - MoleculeNet Lipophilicity set
  (4,200 octanol-water logD values), used for the generality check.
- `data/au2c00122_si_002.xlsx` - the 1,202 experimental lanthanide-extraction logD
  measurements and the four prospective ligands, from the Supporting Information of
  Liu et al., *JACS Au* 2022. This file is third-party copyrighted data and is **not**
  redistributed here (it is git-ignored); download it from the original publication to
  reproduce the f-element results.

## Layout

```
data/        source data (see note above on the third-party xlsx)
src/         active-learning core, acquisition strategies, calibration, figures
results/     computed JSON results + logs that back every number in the paper
figures/     paper figures (pdf/png)
paper/       manuscript (main.tex / main.pdf), SI, bibs, cover letter, TOC graphic
submission/  frozen ACS Omega submission package (self-contained)
```

## Reproduce

Run from the repo root (scripts use repo-root-relative paths):

```bash
python src/real_al.py          # AL on the 1,202 f-element logD measurements
python src/lipo_al.py          # generality replication on Lipophilicity (4,200)
python src/al_calibration.py   # split-conformal calibration
python src/prospective.py      # prospective test on the four new ligands
python src/figures_ggplot.py   # regenerate figures into figures/
```

Computations were run on a single Apple M4 laptop.

## Provenance

This repository was split out of `actinide-mlp-inverse-design`, where it originally
lived on the `path-b-active-learning` branch alongside an unrelated actinide
machine-learning-potential study. It is now maintained independently.
