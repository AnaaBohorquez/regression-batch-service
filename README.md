# Wizeline ML challenge - batch regression

Regression model that predicts `target` from 20 numeric features, packaged as a batch
prediction pipeline with a small CLI (`train` / `predict`).

## Status

- [x] Exploratory analysis (`notebooks/01_eda.ipynb`)
- [x] Model selection (`notebooks/02_modeling.ipynb`)
- [x] Training / prediction pipeline (`src/regressor`)
- [x] Tests (`tests/`)
- [x] Docker image (`Dockerfile`)

## Results

| model | CV RMSE (640 rows) | holdout RMSE (160 rows) |
|---|---|---|
| mean baseline | 5.12 | - |
| linear regression | 2.83 | - |
| gradient boosting, 20 features | 2.15 | - |
| gradient boosting, 5 features, tuned | 1.92 | 1.79 (MAE 1.49, R² 0.87) |

- 160 rows were held out before any modeling decision and used once, at the end. Model choice,
  feature selection and tuning were done with 5-fold CV on the remaining 640 rows.
- Only 5 of the 20 features carry signal (`feature_2`, `feature_13`, `feature_9`, `feature_11`,
  `feature_18`). Permutation importance shows a clear gap between them and the rest, and dropping
  the other 15 lowers the CV RMSE from 2.15 to 2.03.
- `feature_18` has almost no linear correlation with the target (0.06) but the model relies on it:
  its relationship with the target is curved, which a correlation coefficient can't capture.
- The chosen model and hyperparameters live in `configs/model.json`. The pipeline reads that file,
  so nothing is copied by hand between the notebook and the code.

## Setup

Python 3.12.

```bash
python -m venv .venv
# Windows: .\.venv\Scripts\Activate.ps1   |   Linux/macOS: source .venv/bin/activate
pip install -r requirements.txt
```

`requirements.txt` installs the libraries for notebooks, pipeline and tests, plus the `regressor`
package itself in editable mode (`-e .`).

Copy `training_data.csv` and `blind_test_data.csv` into `data/`.

## Usage

Train on the full training set (800 rows) with the config from `configs/model.json`:

```bash
python -m regressor train --data data/training_data.csv
```

This writes two files to `artifacts/`:
- `model.joblib`: the fitted model
- `metadata.json`: training date, features, hyperparameters, scikit-learn version, validation
  metrics from the notebook and the range of each feature in the training data

Predict a batch:

```bash
python -m regressor predict --input data/blind_test_data.csv --output predictions/target_pred.csv
```

The output CSV has a single column, `target_pred`, with one row per input row in the same order.
Columns other than the 5 model features are ignored.

Before predicting, each feature of the batch is compared with its training profile. A warning is
logged (the prediction still runs) when more than 5% of the values fall outside the training range
or when the mean moves more than 0.25 training standard deviations.

Optional arguments: `--config` and `--artifacts` (see `python -m regressor train --help`).

## Tests

```bash
python -m pytest
```

19 tests on synthetic data (they don't need the challenge CSVs): config loading, input validation,
train/predict end to end, output format, drift warnings and CLI exit codes.

## Project layout

```
configs/model.json      model, features and hyperparameters chosen in notebook 02
data/                   challenge CSVs (not versioned, see data/README.md)
notebooks/              01_eda, 02_modeling
src/regressor/
    __main__.py         CLI: train / predict
    config.py           reads configs/model.json
    data.py             CSV loading and validation
    training.py         fits the model, saves model.joblib + metadata.json
    inference.py        loads the model, drift check, writes predictions
tests/                  pytest suite
artifacts/              trained model (generated, not versioned)
predictions/            prediction files (generated, not versioned)
```

## Docker

The image contains Python 3.12, the four runtime libraries pinned to exact versions and the
`regressor` package. Data, model and predictions stay on the host and are mounted as volumes,
so the image doesn't need to be rebuilt when the model is retrained.

```bash
docker build -t regressor .

docker run --rm -v "$(pwd)/data:/app/data" -v "$(pwd)/artifacts:/app/artifacts" \
    regressor train --data data/training_data.csv

docker run --rm -v "$(pwd)/data:/app/data" -v "$(pwd)/artifacts:/app/artifacts" \
    -v "$(pwd)/predictions:/app/predictions" \
    regressor predict --input data/blind_test_data.csv --output predictions/target_pred.csv
```

On Windows PowerShell replace `$(pwd)` with `${PWD}` and the trailing `\` with a backtick, or write
each command on a single line.

Train inside the container too (not only predict): the model file must be loaded with the same
scikit-learn version that created it, and the image pins that version.

## Production notes

How the solution would evolve if it were used over time:

- **Scheduled batch predictions.** `predict` runs on each incoming batch (e.g. nightly). The model is
  loaded from `artifacts/`, so predicting never retrains.
- **Monitoring.** Two signals: the drift warnings on every batch, and the real error once the true
  target of past batches becomes available.
- **Retraining with new data.** New labeled rows are appended to the training data and `train` is run
  again. The new model replaces the current one only if it performs better on a common validation set,
  and the previous `model.joblib` + `metadata.json` are kept so a rollback is a file copy.
- **Model versioning.** `metadata.json` already records what each model was trained with. In a real
  deployment the artifacts would go to a model registry or object storage instead of a local folder.
- **Known limitation.** The drift thresholds are fixed numbers. With small batches the mean of a
  feature moves more by chance, so a statistical test that accounts for batch size (e.g.
  Kolmogorov-Smirnov, already used in the EDA) would give fewer false alarms.
