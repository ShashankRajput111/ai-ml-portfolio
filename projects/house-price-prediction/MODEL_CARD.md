# Model Card: House Price Regression Baseline

## Model overview

- **Task:** Regression
- **Target:** `target_house_price`
- **Selected estimator:** Median `DummyRegressor`, wrapped in a preprocessing pipeline
- **Selection method:** Highest mean R² across shuffled five-fold cross-validation on the training partition
- **Status:** Research/learning baseline only

## Intended use

This artifact demonstrates a reproducible tabular machine-learning workflow: schema checking, missing-value handling, categorical encoding, model comparison, holdout evaluation, model serialization, and batch inference.

It is not suitable for lending, appraisal, investment, taxation, insurance, or other real-world decisions.

## Data

The supplied CSV has 1,200 rows, nine input features, and one regression target. Missing values occur in `income`, `education_level`, `credit_score`, and `loan_amount`. The `city_index` value `?` is treated as missing.

The original file did not include source, geography, currency, time period, collection method, or license information. Those omissions materially limit interpretation and public reuse.

## Preprocessing

- Numeric columns: median imputation, missingness indicators, standard scaling.
- Categorical columns: most-frequent imputation, missingness indicators, one-hot encoding with unknown-category handling.
- All preprocessing is learned inside each training fold to avoid leakage.

## Evaluation

The project reserves 20% of rows as a deterministic holdout (`random_state=42`) and compares four candidates on the remaining rows using five-fold cross-validation. Exact values are in `models/metrics.json`.

The selected median baseline scores approximately:

- Holdout R²: **-0.002**
- Holdout MAE: **66,651** target units
- Holdout RMSE: **91,310** target units

A negative R² means the model is not better than predicting the holdout mean. Feature-based candidates also fail to beat the baseline, so a more complex estimator is not justified.

## Limitations and risks

- The saved estimator predicts a constant median and should not be represented as a useful pricing system.
- The available variables do not include core property attributes such as floor area, bedrooms, bathrooms, property type, condition, precise location, or sale date.
- Target values include extreme and negative observations. They are retained because no authoritative cleaning rule was supplied.
- Missing provenance prevents checks for sampling bias, label quality, temporal leakage, fairness, and market drift.
- Joblib files can execute code when loaded; only load artifacts from trusted sources.

## Recommended next step

Collect a documented property-sales dataset with relevant property attributes, location and time fields, a clear currency, and a trustworthy sale-price label. Define acceptable data ranges with domain input before removing observations. Re-run this repository's validation workflow and require a positive, stable cross-validated R² that materially exceeds the median baseline before considering deployment.
