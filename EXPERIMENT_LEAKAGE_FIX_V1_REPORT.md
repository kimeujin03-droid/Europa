# Leakage Fix v1 Experiment Report

This report records the first leakage-control rerun after the library-informed hybrid endmember update.

## Benchmark Version

- Tag: `leakage_fix_v1_logreg`
- Dataset: `data/processed/synthetic_dataset.csv`
- Endmember basis: library-informed hybrid endmember set
- Main model: additive logistic regression
- Synthetic sample size: 8000
- Main context setting: `rho_geo = 0.75`, `rho_rad = 0.75`
- Seeds for Experiment 1 and ambiguous subset: 20

## Code Changes

### A1. Mixture leakage control

The `simple_organic` mixture range was made deliberately overlapping between the positive `ocean_organic` class and the hard-negative `radiation_mimic` class. Radiation mimics were also given more overlapping `ocean_salt` contribution.

This prevents a simple rule such as "higher organic amount implies positive" from driving the benchmark.

### Mixture weight audit

The generated dataset now stores normalized mixture weights:

- `w_ice`
- `w_ocean_salt`
- `w_simple_organic`
- `w_tholin_pah`
- `w_sulfuric_acid_hydrate`
- `w_sulfur_so2`
- `w_h2o2`
- `w_rad_salt`

Audit outputs:

- `results/qc/mixture_weight_summary.csv`
- `results/qc/mixture_weight_overlap_pairs.csv`
- `results/qc/mixture_weight_overlap_boxplot.png`

Key overlap results:

| Pair | Weight | Overlap coefficient |
| --- | ---: | ---: |
| ocean_organic vs radiation_mimic | `w_simple_organic` | 0.5685 |
| ocean_organic vs exogenic_complex_organic | `w_simple_organic` | 0.6437 |
| radiation_mimic vs exogenic_complex_organic | `w_simple_organic` | 0.8790 |
| ocean_organic vs radiation_mimic | `w_ocean_salt` | 0.1995 |

### A2. Endmember band correction

The sulfuric-acid-hydrate literature-shape proxy was adjusted away from the strongest water-ice degeneracy bands and toward a broader shape-constrained radiolysis template:

- 1.36 um
- 1.79 um
- 2.10 um
- 3.05 um
- 3.90 um
- 4.45 um

The H2O2 proxy was reduced to a single weak 3.50 um marker. The old auxiliary 2.85 um feature was removed.

The toy fallback simple-organic 2.28 um feature and toy H2O2 2.8 um feature were also removed to avoid accidental fallback leakage.

## QC Results

Processed endmember coverage check passed for all eight endmembers:

- wavelength range: 0.7-5.2 um
- NaN count: 0 for all endmembers

Endmember similarity QC showed the expected derived-proxy behavior:

- `ocean_salt` vs `rad_salt_proxy`: Pearson r = 0.9782, spectral angle = 4.081 deg
- `sulfuric_acid_hydrate` vs `rad_salt_proxy`: Pearson r = 0.9752, spectral angle = 8.516 deg

This is expected because `rad_salt_proxy` is a derived mixture proxy, not an independent laboratory endmember.

Context VIF QC passed:

- max VIF = 3.56
- no context feature exceeded VIF 5

## Experiment 1: Main Benchmark

File: `results/experiment1_metrics_leakage_fix_v1_logreg.csv`

| Input setting | PR-AUC | Precision@10% | Recall@10% | Brier |
| --- | ---: | ---: | ---: | ---: |
| Spectral-only | 0.8951 +/- 0.0104 | 0.9625 +/- 0.0131 | 0.5342 +/- 0.0151 | 0.0705 +/- 0.0042 |
| Spectral + geology | 0.9283 +/- 0.0079 | 0.9871 +/- 0.0081 | 0.5479 +/- 0.0162 | 0.0573 +/- 0.0046 |
| Spectral + radiation | 0.9006 +/- 0.0097 | 0.9669 +/- 0.0125 | 0.5366 +/- 0.0152 | 0.0681 +/- 0.0040 |
| Full spatial-spectral | 0.9299 +/- 0.0075 | 0.9881 +/- 0.0068 | 0.5484 +/- 0.0154 | 0.0563 +/- 0.0046 |
| Context-only baseline | 0.4584 +/- 0.0232 | 0.4594 +/- 0.0344 | 0.2550 +/- 0.0208 | 0.1461 +/- 0.0064 |

Interpretation:

The full model improves PR-AUC over spectral-only by about 0.035 while remaining far above context-only. This supports the claim that the model is not simply using spatial priors to label candidates.

Top-10 radiation mimic contamination remained near zero for both spectral-only and full model, so this run should not be framed as a dramatic global radiation-FPR reduction result. The stronger claim is improved ranking and controlled context-sensitive behavior in ambiguous cases.

## Experiment 2: Prior-Strength Sweep

File: `results/prior_sweep_metrics_leakage_fix_v1_logreg.csv`

Summary over the 5x5 grid:

- mean delta PR-AUC, full minus spectral-only: +0.0280
- mean delta Recall@10%: +0.0143
- mean full minus context-only PR-AUC gap: +0.4855

Corner case:

At `rho_geo = 0`, `rho_rad = 0`, the full model and spectral-only model were nearly identical:

- spectral-only PR-AUC = 0.8888
- full PR-AUC = 0.8906
- delta PR-AUC = +0.0018

This is the key arbitrariness defense: when class and spatial context are decorrelated, spatial context gives almost no benefit.

## Experiment 3: Same-Spectrum Stress Test

File: `results/same_spectrum_paired_stats_leakage_fix_v1_logreg.csv`

The test selected spectra whose spectral-only score was ambiguous (`0.4 <= score <= 0.7`) and duplicated each spectrum under exchange-favorable and radiation-prone contexts.

Results:

- spectral-only paired delta mean = 0.0000
- full paired delta mean = 0.8153
- full paired delta std = 0.1658
- sign-test p = 1.58e-30

Interpretation:

This is a controlled behavioral probe, not real Europa validation. It shows that for the same ambiguous spectrum, the full model systematically changes triage score under deliberately contrasting spatial contexts.

## Ambiguous Subset Evaluation

File: `results/ambiguous_subset_metrics_leakage_fix_v1_logreg.csv`

The subset contained validation samples whose spectral-only score was ambiguous (`0.4 <= score <= 0.7`).

| Input setting | PR-AUC | Precision@10% | Recall@10% | Brier |
| --- | ---: | ---: | ---: | ---: |
| Spectral-only | 0.3090 +/- 0.0417 | 0.3090 +/- 0.0950 | 0.1291 +/- 0.0396 | 0.2766 +/- 0.0082 |
| Full spatial-spectral | 0.5644 +/- 0.0786 | 0.6306 +/- 0.1310 | 0.2633 +/- 0.0565 | 0.2111 +/- 0.0194 |
| Context-only baseline | 0.3137 +/- 0.0550 | 0.2700 +/- 0.1016 | 0.1125 +/- 0.0442 | 0.3719 +/- 0.0218 |

Interpretation:

The ambiguous subset is where the spatial-spectral model has the clearest value. Full improves PR-AUC, Precision@10%, Recall@10%, and Brier score relative to spectral-only, while context-only remains poor.

## Main Outputs

- `data/processed/synthetic_dataset.csv`
- `data/processed/endmembers/*.csv`
- `results/experiment1_metrics_leakage_fix_v1_logreg.csv`
- `results/experiment1_metrics_by_seed_leakage_fix_v1_logreg.csv`
- `results/experiment1_predictions_leakage_fix_v1_logreg.csv`
- `results/experiment1_pr_curve_leakage_fix_v1_logreg.png`
- `results/prior_sweep_metrics_leakage_fix_v1_logreg.csv`
- `results/prior_sweep_heatmap_delta_pr_auc_leakage_fix_v1_logreg.png`
- `results/prior_sweep_heatmap_delta_recall10_leakage_fix_v1_logreg.png`
- `results/prior_sweep_heatmap_top10_rad_reduction_leakage_fix_v1_logreg.png`
- `results/same_spectrum_paired_stats_leakage_fix_v1_logreg.csv`
- `results/same_spectrum_paired_deltas_leakage_fix_v1_logreg.csv`
- `results/same_spectrum_scores_leakage_fix_v1_logreg.csv`
- `results/same_spectrum_comparison_leakage_fix_v1_logreg.png`
- `results/ambiguous_subset_metrics_leakage_fix_v1_logreg.csv`
- `results/ambiguous_subset_metrics_by_seed_leakage_fix_v1_logreg.csv`
- `results/ambiguous_subset_predictions_leakage_fix_v1_logreg.csv`
- `results/qc/mixture_weight_summary.csv`
- `results/qc/mixture_weight_overlap_pairs.csv`
- `results/qc/mixture_weight_overlap_boxplot.png`
- `results/qc/endmember_similarity_pairs.csv`
- `results/qc/context_feature_vif.csv`

## Limitations

This benchmark is not a fully laboratory-spectra-only benchmark. It is a library-informed hybrid benchmark combining laboratory/library spectra, literature-constrained shape proxies, functional organic proxies, and derived radiolytic salt proxies.

Therefore, claims should be limited to controlled synthetic triage behavior:

- supported: spatial context improves ranking and ambiguous-case behavior
- supported: context-only does not replace spectral evidence
- supported: prior sweep shows the effect collapses when context is decorrelated
- not supported in this run: dramatic global reduction of radiation-mimic false positives across the full benchmark
