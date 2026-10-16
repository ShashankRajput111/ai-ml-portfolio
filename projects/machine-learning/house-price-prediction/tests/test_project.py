from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import pytest

from src.predict import predict, validate_features
from src.train import (
    DEFAULT_DATA_PATH,
    DEFAULT_MODEL_DIR,
    EXPECTED_FEATURES,
    load_dataset,
    split_features_target,
)


MODEL_PATH = DEFAULT_MODEL_DIR / "house_price_pipeline.joblib"


def test_dataset_has_expected_schema() -> None:
    data = load_dataset(DEFAULT_DATA_PATH)
    X, y = split_features_target(data)

    assert X.columns.tolist() == EXPECTED_FEATURES
    assert len(X) == len(y) == 1200
    assert not y.isna().any()
    assert not X["city_index"].eq("?").any()


def test_saved_model_produces_finite_predictions() -> None:
    data = load_dataset(DEFAULT_DATA_PATH)
    X, _ = split_features_target(data)
    model = joblib.load(MODEL_PATH)
    predictions = model.predict(X.head(3))

    assert predictions.shape == (3,)
    assert np.isfinite(predictions).all()


def test_prediction_wrapper_preserves_rows() -> None:
    row = pd.DataFrame(
        [
            {
                "age": 45,
                "income": 75_000,
                "experience_years": 18,
                "education_level": "bachelor",
                "hours_per_week": 40,
                "city_index": "B",
                "credit_score": 720,
                "loan_amount": 25_000,
                "debt_ratio": 0.35,
            }
        ]
    )
    result = predict(row, MODEL_PATH)

    assert len(result) == 1
    assert np.isfinite(result.loc[0, "predicted_house_price"])


def test_missing_feature_is_rejected() -> None:
    incomplete = pd.DataFrame([{"age": 45}])
    with pytest.raises(ValueError, match="missing required columns"):
        validate_features(incomplete)


def test_artifact_paths_are_repository_relative() -> None:
    assert DEFAULT_DATA_PATH.is_file()
    assert MODEL_PATH.is_file()
    assert Path("models/house_price_pipeline.joblib") == MODEL_PATH.relative_to(
        MODEL_PATH.parents[1]
    )
