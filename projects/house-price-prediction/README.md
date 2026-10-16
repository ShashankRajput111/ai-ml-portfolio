# House Price Prediction

A reproducible regression project rebuilt from an exploratory Jupyter notebook. It includes a leakage-safe preprocessing pipeline, model comparison, a trained artifact, a prediction CLI, tests, and continuous integration.

> **Important result:** the supplied dataset does not contain enough predictive signal to beat a median-price baseline. The selected pipeline has a holdout R² of approximately **-0.002**, MAE of **66,651**, and RMSE of **91,310**. This repository is suitable as a transparent learning project, but the current model should **not** be used for real pricing decisions.

## What was fixed

- Corrected the outlier-filtering typo that calculated bounds for `credit_score` but filtered `income`.
- Removed data leakage by fitting imputers, encoders, and scalers only inside training folds.
- Replaced the hard-coded Windows data path with repository-relative paths.
- Replaced deprecated `seaborn.distplot` usage.
- Avoided chained `inplace=True` operations that produce pandas warnings and may stop working.
- Replaced fixed `input_dim=11` values with a preprocessing pipeline that determines its own feature count.
- Added deterministic train/test splitting and five-fold cross-validation.
- Replaced classification-only `accuracy_score` with regression metrics: R², MAE, and RMSE.
- Removed the invalid final call `ann.predict(y_test_predict, y_test)`.
- Added model serialization, metadata, a batch prediction command, tests, and GitHub Actions.

## Why the ANN was replaced

The original neural network achieved about **-2.64 test R²** (shown as `-264%` in the notebook), meaning it performed much worse than predicting a constant. This dataset has only 1,200 rows and almost no measurable relationship between the available features and the target. A small scikit-learn benchmark is faster, easier to reproduce, and makes the lack of learnable signal visible instead of hiding it behind a larger model.

The current selection process compares a median baseline, Ridge, ElasticNet, and gradient boosting using five-fold cross-validation. Because none of the feature-based models beats the baseline, the saved artifact is intentionally the median baseline. That is the statistically honest result for this data—not a production model.

## Repository structure

```text
.
├── .github/workflows/ci.yml
├── data/
│   ├── README.md
│   └── unclean_regression_dataset.csv
├── models/
│   ├── holdout_predictions.csv
│   ├── house_price_pipeline.joblib
│   ├── metrics.json
│   └── model_metadata.json
├── notebooks/house_price_prediction.ipynb
├── reports/figures/
├── src/
│   ├── predict.py
│   └── train.py
├── tests/test_project.py
├── MODEL_CARD.md
├── requirements.txt
└── requirements-dev.txt
```

## Quick start

Python 3.11 or 3.12 is recommended.

```bash
python -m venv .venv
```

Activate the environment:

```bash
# Windows PowerShell
.venv\Scripts\Activate.ps1

# macOS/Linux
source .venv/bin/activate
```

Install dependencies and retrain:

```bash
python -m pip install -r requirements.txt
python -m src.train
```

Run the notebook:

```bash
jupyter lab notebooks/house_price_prediction.ipynb
```

## Make a prediction

The CLI accepts one JSON object, a JSON list, or a CSV file.

```bash
python -m src.predict --input example_input.json
```

Write batch predictions to a CSV:

```bash
python -m src.predict --input example_input.json --output predictions.csv
```

## Run tests

```bash
python -m pip install -r requirements-dev.txt
python -m pytest
```

## Evaluation

Model choice uses shuffled five-fold cross-validation on the 80% training partition. The untouched 20% holdout is used once for final reporting. After evaluation, the selected pipeline is refit on all 1,200 rows for the distributable artifact.

Exact results are stored in [`models/metrics.json`](models/metrics.json). The negative R² values and weak correlations indicate that the current borrower-style features—age, income, education, credit score, debt, and related variables—do not explain the supplied house-price target. Stronger property features such as location, floor area, bedrooms, bathrooms, property type, condition, and sale date are likely required.

## Known limitations

- The data source and license were not included with the original notebook. Confirm that you have permission before publishing the CSV.
- The target includes extreme values, including a negative price. They are retained because there is no authoritative rule proving they are data-entry errors.
- The feature set describes people and borrowing characteristics more than houses.
- The model has not been validated for any geography, currency, time period, or real-world population.
- A serialized scikit-learn model should only be loaded from a trusted source.

## Publish to GitHub

Create an empty GitHub repository, then run these commands from this folder:

```bash
git init
git add .
git commit -m "Add reproducible house price prediction project"
git branch -M main
git remote add origin https://github.com/YOUR_USERNAME/YOUR_REPOSITORY.git
git push -u origin main
```

Do not publish the dataset until you have confirmed its license and sharing permissions.
