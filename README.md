# XRPL fraud detector (v2) — burst-component features

## Reproduce the predictions
Place these files next to `train/`, `validation/`, `test/`, then:

    pip install -r requirements.txt
    python run_pipeline.py            # writes outputs/predictions_test.csv and predictions_validation.csv

Runs in about 2 minutes on one CPU core (features 10 s, 10 model fits). Python 3.10+.
`run_pipeline.py` uses LightGBM when installed, otherwise scikit-learn's HistGradientBoostingClassifier.
`fraud_detection.ipynb` is the full exploratory notebook (IsolationForest, LightGBM, GNN, blend), with outputs cleared; re-run it to regenerate them.

## What changed vs. v1
`xrpl_features.py` adds ~108 label-free features: burst components (windows 60/120/300 s) and account-relative windows (±5/15/60 min).
Grouped 5-fold out-of-fold results on `train/` (scenario/wallet groups, no labels used as features):

| | AP x100 | P@1% | R@1% |
|---|---|---|---|
| v1 features (HistGradientBoosting) | 50.1 | 0.77 | 0.25 |
| v2 features (HistGradientBoosting) | 56.1 | 0.82 | 0.26 |

Recall@1% cannot exceed ~0.32 (top 1% = ~401 rows vs 1,250 positives). Medium-tier AP is ~5 and hard-tier ~0.7 (the relay hubs and 25 ambiguous rows are statistically indistinguishable from normals with the available labels).
Tried and rejected: windows >= 600 s (giant-component percolation, train/test drift), second-stage stacking (lower AP).

## Not written by us
Libraries: pandas, numpy, scipy, scikit-learn, LightGBM, PyTorch, matplotlib, seaborn. Starter notebook/`payments.py` from the organisers. AI assistance: Claude (Anthropic) helped design and write the feature code and analysis.
