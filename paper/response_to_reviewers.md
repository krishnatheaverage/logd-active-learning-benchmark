# Response to Reviewers — ACS Omega ao-2026-07482d

**Title:** Objective-Dependent Active Learning and Calibrated Uncertainty for Sample-Efficient Discovery of Rare-Earth Extractants: A Benchmark on 1,202 Experimental Distribution Coefficients with Prospective Validation
**Author:** Krishna Harish

We thank the reviewer for a careful and constructive report. Every point has been
addressed with **new experiments on the real data** (no claim is asserted without a
computed result), and the manuscript has been substantially expanded: it now reports
**five findings** (adding a leakage-controlled generalization analysis and a
definition-dependent discovery analysis), **four new tables**, and **five new
figures**, all reproduced by released code. Below, each reviewer comment is quoted
and followed by our response and the corresponding change. Section, table, and figure
numbers refer to the revised manuscript. All new code is in the `revision/` directory
of the released repository and regenerates every number here.

## Summary of major changes
1. **Leakage-controlled splits** (new Fig. 1, Table 1): random row-level CV is shown
   to be strongly optimistic; ligand-, source-, and structural-family-disjoint splits
   collapse R² from 0.88 to ~0.2 or below. Directly reframes finding (i).
2. **Ligand-batch acquisition** (new Fig. 4): a chemically realistic decision unit
   (select a ligand, measure all its conditions); the objective-dependent dichotomy
   persists.
3. **Definition of a "top" system** (new Fig. 5, Table 2): recall reported for rows,
   ligand–metal pairs, and unique ligands; the strategy ranking inverts across them.
4. **Rigorous, subgroup-resolved calibration** (new Fig. 6, Table 3): ligand-disjoint
   train/calibration/test; coverage by logD region and lanthanide class; Mondrian
   (group-conditional) conformal; and a demonstration that non-monotone calibration
   reorders the uncertainty ranking (so it can change acquisition).
5. **Broader model/UQ baselines** (new Fig. 7, Table 4): random forest, quantile
   regression forest, Gaussian process, gradient boosting, MC-dropout Bayesian NN,
   with uncertainty-ranking-quality and enrichment metrics.
6. **Expanded literature** (17 new references, 2013–2025) positioning the work.

---

## Reviewer 1 (Recommendation: publish after major revisions)

We are grateful for the positive assessment of the objective-dependent framing, the
random-forest baseline, the multi-seed curves, the independent-dataset replication,
and the prospective test.

### 1.1 — Split rigor / data leakage
> "If train/test splits are made at the row level, closely related measurements from the same ligand or condition family can appear in both training and testing, potentially inflating validation performance. A more stringent split by ligand scaffold, publication source, or ligand family would better represent discovery of genuinely new extractants... scaffold level validation would provide a more realistic estimate of deployability."

**Response.** We agree, and this is now a central result. Although the released feature
matrix contains no SMILES (so exact Bemis–Murcko scaffolds are not recoverable), the
grouping structure is: identical ECFP bit-vectors identify the **93 unique ligands**,
the metal-descriptor block identifies the **14 lanthanides**, and the citation index
identifies the **45 publication sources**. We added leakage-controlled evaluation under
four regimes (new §"Realistic generalisation…", **Fig. 1**, **Table 1**):

| Split | RF R² |
|---|---|
| Random (row-level) | 0.88 ± 0.03 |
| Ligand-disjoint | 0.20 ± 0.31 |
| Source-disjoint | −0.03 ± 0.80 |
| Ligand-family (ECFP-Tanimoto, scaffold-split proxy) | −0.24 ± 0.50 |
| Prospective (4 new ligands) | 0.70 |

The row-level R² is strongly optimistic; grouped splits collapse it to at or below the
mean. We now state explicitly that in-distribution R² is **not** a deployability
estimate, that realistic generalization to new chemotypes is much lower, and that the
prospective R²=0.70 sits between the extremes because the four new ligands are close
analogues of training chemotypes. We cite Sheridan 2013, Guo et al. 2024/2025 on
split optimism. Because SMILES are unavailable, we are transparent that the family
split is an ECFP-Tanimoto proxy for a scaffold split (noted in Limitations).

### 1.2 — Experimental decision unit / batch acquisition
> "Selecting one ligand can imply measuring multiple lanthanides, acidities, diluents, or concentrations... Ranking and acquiring individual rows can therefore overstate operational flexibility. A ligand batch acquisition simulation... would make the benchmark more chemically realistic."

**Response.** Added (new §"…ligand-batch unit", **Fig. 4**). We re-ran the entire
benchmark with a **ligand-batch acquisition unit**: selecting a ligand reveals **all**
of its measurements (median 14, across lanthanides/acidities/diluents/temperatures),
and the test set is a **ligand-disjoint** hold-out. Result: the objective-dependent
ordering persists under this realistic unit — exploitation gives the best top-10
**ligand** recall (1.00) while random/exploration lags (0.83); on prediction, all R²
are low under the hard ligand-disjoint test (consistent with 1.1), random giving the
best model (R² = 0.30) by a small margin over exploitation (0.29). We report this
honestly: the design-side dichotomy is clear and the ordering is preserved, while the
prediction-side gap narrows once train–test ligand overlap is forbidden. So the
finding is not an artifact of scoring isolated rows.

### 1.3 — Definition of "top-10 recall"
> "The 'top 10 recall' metric should specify whether 'top' systems are unique ligands, ligand–metal pairs, or condition specific measurements; discovery value differs substantially across these definitions."

**Response.** Added (new §"What counts as a 'top' system", **Fig. 5**, **Table 2**). We
now report recall under all three definitions, and the ranking of strategies
**inverts**:

| Strategy | recall (row) | recall (ligand–metal) | recall (ligand) |
|---|---|---|---|
| random | 0.45 | 0.72 | 0.95 |
| greedy (exploit) | 1.00 | 0.88 | 0.82 |
| EI | 0.93 | 0.90 | 0.97 |

Exploitation dominates at the **row** level (targets exact high-logD conditions) but at
the **ligand** level broad sampling already recovers most top ligands (each ligand
appears under many conditions, so it is easy to hit without targeting). We therefore
report all three definitions and caution that a single unstated "top-k recall" can
reverse the practical conclusion.

### 1.4 — Conformal rigor and subgroup coverage
> "The conformal procedure needs more precise reporting. Calibration splits should be fully separated from both training and final testing... Reporting coverage by chemical subgroup, logD range, lanthanide identity, and prospective ligands would be more informative than global coverage alone... non monotonic or locally adaptive calibration could alter rankings."

**Response.** Fully reworked (new §"Calibration matters for trust…", **Fig. 6**,
**Table 3**). (a) Train/calibration/test are now **ligand-disjoint**, so no ligand
leaks into the rescaling. (b) We report coverage of nominal-80% intervals **by
subgroup**: under the realistic split, raw intervals **under**-cover (0.72), a single
global conformal factor restores average coverage (0.90) but leaves the high-logD
region uneven, and **Mondrian** (group-conditional) conformal (0.87) evens the
subgroups (by-logD coverage low/mid/high: global 0.87/0.92/0.87 vs Mondrian
0.96/0.82/0.88; per-lanthanide-class coverage 0.89–0.91). Prospective-ligand coverage
is 0.875. (c) Crucially, we **quantify the non-monotonic caveat**: Mondrian reorders
the uncertainty ranking versus the global factor (Spearman ρ = 0.85 < 1), so locally
adaptive calibration **can** change AL acquisition even though a rank-preserving global
factor cannot. We retain the in-distribution result (raw calibration error 0.103 →
0.013 under split-conformal on the row split) and now frame the two as complementary
(in-distribution vs. distribution-shift).

### 1.5 — Broader baselines and metrics
> "Gaussian process regression, quantile random forests, gradient boosting, Bayesian neural networks, and conformalized residual models could provide more insight into uncertainty ranking quality. Reporting top-k enrichment, regret, hit rate above a practical logD threshold, and Pareto performance... would strengthen the connection to separations chemistry."

**Response.** Added a five-estimator comparison (new §"Broader model…", **Fig. 7**,
**Table 4**): random forest, **quantile regression forest**, **Gaussian process**
(PCA-reduced), **gradient boosting**, and an **MC-dropout Bayesian NN**, each with a
split-conformal (conformalized-residual) wrapper, on a ligand-disjoint split. We score
predictive accuracy, conformal coverage, and **uncertainty-ranking quality** (Spearman
of predicted σ vs. realized error; selective-prediction RMSE-AUC). We also added the
requested **discovery metrics** to the AL benchmark: **enrichment factor**, **simple
regret**, and **hit-rate above logD ≥ 1** (D ≥ 10) (Table 2), and the R²-vs-recall
**Pareto** view (Figs. 3–5). Summary: no estimator uniformly dominates (consistent
with Hirschfeld 2020, Scalia 2020), with high fold-to-fold variance on this hard
ligand-disjoint split; gradient boosting is the most consistent and has the best mean
uncertainty-ranking quality (ρ = 0.28), while the Gaussian process is weakest. The
dichotomy and the calibration-vs-acquisition distinction hold across estimators, i.e.
they are properties of pool-based AL, not of the random forest.

On the reviewer's point that "a high distribution coefficient alone is not always
sufficient," we added a **selectivity analysis** (new subsection + figure). Per ligand
we compute the separation factor as the spread of logD across the 14 lanthanides
(range up to 6.1 log units over 63 ligands). Ranking ligands by extraction strength
(max logD) is **not** the same as ranking by selectivity (Spearman 0.63); only **4 of
the top-10** strongest extractants are also among the top-10 most selective, and we
give the strength–selectivity Pareto front. Optimizing logD alone therefore misses
most discriminating ligands. A full selectivity-aware acquisition loop is noted as a
natural extension.

### 1.6 / 1.7 — More figures and tables
**Response.** The revision adds **five figures** (splits, ligand-batch, recall
definitions, subgroup calibration, model/UQ comparison) and **four tables** (splits,
discovery metrics, subgroup calibration, model/UQ), in addition to the original three
figures. All are auto-generated from the results JSONs by released scripts.

### 1.8 — More literature / references not up to date
**Response.** The introduction and discussion now situate the work against 17 added,
verified references (2013–2025), including pool-based/batch AL
(Graff 2021, Rohr 2020, Reker 2020, Bayley 2024, Bailey 2023), explore/exploit
(De Ath 2021), realistic splitting (Sheridan 2013, Guo 2024/2025), group-conditional
conformal (Sun 2017), early-recognition metrics (Truchon & Bayly 2007, Ash 2022),
UQ-estimator comparisons (Hirschfeld 2020, Scalia 2020, Soleimany 2021, Busk 2022,
Gal 2016), and current rare-earth logD/extraction ML (Udawattha & Alam 2025,
Chaube 2020, Edaugal 2025).

### Additional questions (technical validity / conclusions supported / references)
The new leakage-controlled evaluation, ligand-batch acquisition, subgroup calibration,
and multi-estimator comparison directly address the "technical quality" and
"conclusions supported" concerns by testing every claim under realistic, non-leaking
conditions and across models; the reference list is now current. We believe the
revised manuscript is substantially strengthened and hope the reviewer agrees.
