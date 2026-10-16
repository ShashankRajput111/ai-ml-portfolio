"""Run batch predictions with the saved house-price pipeline."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from src.train import EXPECTED_FEATURES, PROJECT_ROOT


DEFAULT_MODEL_PATH = PROJECT_ROOT / "models" / "house_price_pipeline.joblib"


def load_input(input_path: Path) -> pd.DataFrame:
    if input_path.suffix.lower() == ".csv":
        return pd.read_csv(input_path)
    if input_path.suffix.lower() == ".json":
        payload = json.loads(input_path.read_text(encoding="utf-8"))
        if isinstance(payload, dict):
            payload = [payload]
        if not isinstance(payload, list):
            raise ValueError("JSON input must be an object or a list of objects.")
        return pd.DataFrame(payload)
    raise ValueError("Input must be a .csv or .json file.")


def validate_features(data: pd.DataFrame) -> pd.DataFrame:
    missing_columns = sorted(set(EXPECTED_FEATURES).difference(data.columns))
    if missing_columns:
        raise ValueError(f"Input is missing required columns: {missing_columns}")
    clean_data = data[EXPECTED_FEATURES].copy()
    clean_data["city_index"] = clean_data["city_index"].replace("?", np.nan)
    return clean_data


def predict(input_data: pd.DataFrame, model_path: Path = DEFAULT_MODEL_PATH) -> pd.DataFrame:
    model = joblib.load(model_path)
    features = validate_features(input_data)
    output = input_data.copy()
    output["predicted_house_price"] = model.predict(features)
    return output


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True, help="Input CSV or JSON.")
    parser.add_argument(
        "--model",
        type=Path,
        default=DEFAULT_MODEL_PATH,
        help="Trained .joblib pipeline.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        help="Optional output CSV. Prints JSON when omitted.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    result = predict(load_input(args.input), args.model)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        result.to_csv(args.output, index=False)
        print(f"Saved {len(result)} prediction(s) to {args.output}")
    else:
        print(result.to_json(orient="records", indent=2))


if __name__ == "__main__":
    main()
