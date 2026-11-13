# Wizeline ML challenge - batch regression

Regression model that predicts `target` from 20 numeric features, packaged as a batch
prediction pipeline (CLI + Docker).

## Status

- [x] Exploratory analysis (`notebooks/01_eda.ipynb`)
- [x] Model selection (`notebooks/02_modeling.ipynb`)
- [ ] Training / prediction pipeline (`src/regressor`)
- [ ] Tests
- [ ] Docker image

## Results so far

| model | CV RMSE (640 rows) | holdout RMSE (160 rows) |
|---|---|---|
| mean baseline | 5.12 | - |
| linear regression | 2.83 | - |
| gradient boosting, 20 features | 2.15 | - |
| gradient boosting, 5 features, tuned | 1.92 | 1.79 (R² 0.87) |

Only 5 of the 20 features carry signal (`feature_2`, `feature_13`, `feature_9`, `feature_11`,
`feature_18`). The chosen model and hyperparameters are stored in `configs/model.json`, which
the pipeline reads at training time.

## Project layout

```
configs/model.json   model, features and hyperparameters chosen in notebook 02
data/                challenge CSVs (not versioned, see data/README.md)
notebooks/           01_eda, 02_modeling
requirements.txt     dependencies for notebooks, pipeline and tests
```

## Setup

Python 3.12.

```bash
python -m venv .venv
# Windows: .\.venv\Scripts\Activate.ps1   |   Linux/macOS: source .venv/bin/activate
pip install -r requirements.txt
```

Copy `training_data.csv` and `blind_test_data.csv` into `data/`.
