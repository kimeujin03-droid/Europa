# Europa Library-Informed Synthetic Benchmark

This repository contains starter code, selected endmember files, QC outputs, and paper-facing CSV/PNG results for a Europa surface spectral triage benchmark.

The benchmark evaluates whether spatial geology/radiation context improves triage of ambiguous ocean-organic candidates relative to a spectral-only baseline. It is not a claim of biosignature detection on Europa.

## Current Benchmark Version

The committed results use a `library-informed hybrid` endmember set:

- USGS H2O ice 77 K plus a long-wavelength proxy tail.
- RELAB magnesium sulfate hydrate for the hydrated salt component.
- USGS benzanthracene as a PAH-like hard-negative organic template.
- USGS sulfur reagent as an auxiliary sulfur radiolysis component.
- Literature-constrained sulfuric-acid-hydrate and H2O2 marker proxies.
- Explicitly labeled functional organic and derived radiolytic salt proxies.

See `data/manifest/endmember_selection.csv` and `data/manifest/endmember_data_status.csv` for source tiers and caveats.

## Main Scripts

```bash
python scripts/05_convert_selected_endmembers.py
python scripts/07_plot_processed_endmembers.py
python scripts/09_qc_collinearity.py
python scripts/10_qc_context_vif.py

python scripts/01_generate_dataset.py --n 8000 --rho-geo 0.75 --rho-rad 0.75
python scripts/02_run_experiment1.py --model logreg --seeds 20 --tag literature_radiation_v1_logreg
python scripts/03_prior_sweep.py --model logreg --n 2500 --tag literature_radiation_v1_logreg
python scripts/04_same_spectrum_test.py --model logreg --tag literature_radiation_v1_logreg
python scripts/13_ambiguous_subset_eval.py --model logreg --seeds 20 --tag literature_radiation_v1_logreg
```

## Key Outputs

- `results/experiment1_metrics_literature_radiation_v1_logreg.csv`
- `results/prior_sweep_metrics_literature_radiation_v1_logreg.csv`
- `results/same_spectrum_paired_stats_literature_radiation_v1_logreg.csv`
- `results/ambiguous_subset_metrics_literature_radiation_v1_logreg.csv`
- `results/processed_endmembers.png`
- `results/paper/`

## Data Policy

Large downloaded archives and broad RELAB metadata caches are excluded from git. The repository includes only small selected source files needed by the current endmember selection plus processed endmember CSV files.
