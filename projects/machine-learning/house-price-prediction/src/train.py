"""Train and evaluate reproducible house-price regression pipelines."""

from __future__ import annotations

import argparse
import json
import platform
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
import sklearn
from sklearn.base import clone
from sklearn.compose import ColumnTransformer
from sklearn.dummy import DummyRegressor
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import ElasticNet, Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import KFold, cross_validate, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DATA_PATH = PROJECT_ROOT / "data" / "unclean_regression_dataset.csv"
DEFAULT_MODEL_DIR = PROJECT_ROOT / "models"
TARGET_COLUMN = "target_house_price"
RANDOM_STATE = 42
TEST_SIZE = 0.20
EXPECTED_FEATURES = [
    "age",
    "income",
    "experience_years",
    "education_level",
    "hours_per_week",
    "city_index",
    "credit_score",
    "loan_amount",
    "debt_ratio",
]


def load_dataset(data_path: Path) -> pd.DataFrame:
    """Load the CSV and normalize the explicit missing-category marker."""
    data = pd.read_csv(data_path)
    required_columns = set(EXPECTED_FEATURES + [TARGET_COLUMN])
    missing_columns = sorted(required_columns.difference(data.columns))
    if missing_columns:
        raise ValueError(f"Dataset is missing required columns: {missing_columns}")
    if data[TARGET_COLUMN].isna().any():
        raise ValueError("Target column contains missing values.")

    data = data[EXPECTED_FEATURES + [TARGET_COLUMN]].copy()
    data["city_index"] = data["city_index"].replace("?", np.nan)
    return data


def split_features_target(data: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series]:
    return data[EXPECTED_FEATURES].copy(), data[TARGET_COLUMN].copy()


def build_preprocessor(features: pd.DataFrame) -> ColumnTransformer:
    """Create preprocessing learned only from each training fold."""
    numeric_features = features.select_dtypes(include="number").columns.tolist()
    categorical_features = [
        column for column in EXPECTED_FEATURES if column not in numeric_features
    ]

    numeric_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median", add_indicator=True)),
            ("scaler", StandardScaler()),
        ]
    )
    categorical_pipeline = Pipeline(
        steps=[
            (
                "imputer",
                SimpleImputer(strategy="most_frequent", add_indicator=True),
            ),
            (
                "onehot",
                OneHotEncoder(handle_unknown="ignore", sparse_output=False),
            ),
        ]
    )

    return ColumnTransformer(
        transformers=[
            ("numeric", numeric_pipeline, numeric_features),
            ("categorical", categorical_pipeline, categorical_features),
        ]
    )


def candidate_models() -> dict[str, Any]:
    """Small, interpretable comparison suitable for this 1,200-row dataset."""
    return {
        "Dummy median baseline": DummyRegressor(strategy="median"),
        "Ridge": Ridge(alpha=10.0),
        "ElasticNet": ElasticNet(
            alpha=1.0,
            l1_ratio=0.2,
            max_iter=20_000,
        ),
        "Gradient boosting": GradientBoostingRegressor(
            random_state=RANDOM_STATE,
            n_estimators=300,
            learning_rate=0.03,
            max_depth=2,
            loss="huber",
        ),
    }


def make_pipeline(features: pd.DataFrame, model: Any) -> Pipeline:
    return Pipeline(
        steps=[
            ("preprocessor", build_preprocessor(features)),
            ("model", model),
        ]
    )


def evaluate_candidates(
    X_train: pd.DataFrame,
    y_train: pd.Series,
) -> tuple[pd.DataFrame, dict[str, Pipeline]]:
    """Score candidates with shuffled five-fold CV on the training partition."""
    cv = KFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
    fitted_templates: dict[str, Pipeline] = {}
    records: list[dict[str, float | str]] = []

    for model_name, model in candidate_models().items():
        pipeline = make_pipeline(X_train, model)
        scores = cross_validate(
            pipeline,
            X_train,
            y_train,
            cv=cv,
            scoring=("r2", "neg_mean_absolute_error", "neg_root_mean_squared_error"),
            n_jobs=1,
        )
        records.append(
            {
                "model": model_name,
                "cv_r2_mean": float(scores["test_r2"].mean()),
                "cv_r2_std": float(scores["test_r2"].std()),
                "cv_mae": float(-scores["test_neg_mean_absolute_error"].mean()),
                "cv_rmse": float(
                    -scores["test_neg_root_mean_squared_error"].mean()
                ),
            }
        )
        fitted_templates[model_name] = pipeline

    results = pd.DataFrame(records).sort_values(
        by=["cv_r2_mean", "cv_mae"],
        ascending=[False, True],
        ignore_index=True,
    )
    return results, fitted_templates


def regression_metrics(
    y_true: pd.Series | np.ndarray,
    y_pred: np.ndarray,
) -> dict[str, float]:
    return {
        "r2": float(r2_score(y_true, y_pred)),
        "mae": float(mean_absolute_error(y_true, y_pred)),
        "rmse": float(mean_squared_error(y_true, y_pred) ** 0.5),
    }


def train_and_save(
    data_path: Path = DEFAULT_DATA_PATH,
    model_dir: Path = DEFAULT_MODEL_DIR,
) -> dict[str, Any]:
    """Select by CV, evaluate once on holdout data, then refit on all rows."""
    data = load_dataset(data_path)
    X, y = split_features_target(data)
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE,
    )

    cv_results, templates = evaluate_candidates(X_train, y_train)
    selected_model_name = str(cv_results.loc[0, "model"])
    evaluation_pipeline = clone(templates[selected_model_name])
    evaluation_pipeline.fit(X_train, y_train)
    test_predictions = evaluation_pipeline.predict(X_test)
    test_metrics = regression_metrics(y_test, test_predictions)

    # After the untouched holdout evaluation, refit the selected pipeline on all rows
    # for the distributable artifact.
    deployment_pipeline = clone(templates[selected_model_name])
    deployment_pipeline.fit(X, y)

    model_dir.mkdir(parents=True, exist_ok=True)
    model_path = model_dir / "house_price_pipeline.joblib"
    metrics_path = model_dir / "metrics.json"
    metadata_path = model_dir / "model_metadata.json"
    predictions_path = model_dir / "holdout_predictions.csv"

    joblib.dump(deployment_pipeline, model_path)

    metrics: dict[str, Any] = {
        "dataset_rows": int(len(data)),
        "training_rows": int(len(X_train)),
        "test_rows": int(len(X_test)),
        "random_state": RANDOM_STATE,
        "test_size": TEST_SIZE,
        "selected_model": selected_model_name,
        "selection_rule": "Highest mean five-fold cross-validation R-squared",
        "cross_validation": cv_results.to_dict(orient="records"),
        "holdout": test_metrics,
        "deployment_fit_rows": int(len(X)),
    }
    metrics_path.write_text(json.dumps(metrics, indent=2), encoding="utf-8")

    metadata = {
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "python_version": platform.python_version(),
        "pandas_version": pd.__version__,
        "numpy_version": np.__version__,
        "scikit_learn_version": sklearn.__version__,
        "joblib_version": joblib.__version__,
        "target": TARGET_COLUMN,
        "features": EXPECTED_FEATURES,
        "missing_value_policy": {
            "city_index_question_mark": "Converted to missing",
            "numeric": "Median imputation learned on training data",
            "categorical": "Most-frequent imputation learned on training data",
        },
        "status": "Research baseline only; holdout R-squared is not positive",
    }
    metadata_path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")

    holdout_predictions = X_test.copy()
    holdout_predictions["actual_house_price"] = y_test
    holdout_predictions["predicted_house_price"] = test_predictions
    holdout_predictions["residual"] = y_test - test_predictions
    holdout_predictions.to_csv(predictions_path, index=False)

    return metrics


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--data",
        type=Path,
        default=DEFAULT_DATA_PATH,
        help="Input CSV path.",
    )
    parser.add_argument(
        "--model-dir",
        type=Path,
        default=DEFAULT_MODEL_DIR,
        help="Directory for the trained model and evaluation files.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    metrics = train_and_save(args.data, args.model_dir)
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
