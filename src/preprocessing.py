"""Shared schema and validation for training and inference."""
from typing import Any, Mapping
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder

# Demo nutrient units are mg/kg; real datasets must use the same units.
RANGES = {
    "Nitrogen": (0.0, 500.0), "Phosphorus": (0.0, 500.0),
    "Potassium": (0.0, 500.0), "Temperature": (-10.0, 60.0),
    "Humidity": (0.0, 100.0), "Moisture": (0.0, 100.0), "pH": (0.0, 14.0),
}
NUMERIC = list(RANGES)
CATEGORICAL = ["Crop", "Soil_Type"]
FEATURES = NUMERIC + CATEGORICAL
TARGET = "Fertilizer"


def validate_frame(frame: pd.DataFrame, training: bool = False) -> pd.DataFrame:
    """Return a validated copy; reject missing, nonfinite and out-of-range values."""
    required = FEATURES + ([TARGET] if training else [])
    missing = set(required) - set(frame.columns)
    if missing:
        raise ValueError("Missing required columns: " + ", ".join(sorted(missing)))
    if frame.empty:
        raise ValueError("The dataset contains no rows.")
    result = frame[required].copy()
    for column, (low, high) in RANGES.items():
        if result[column].map(lambda value: isinstance(value, (bool, np.bool_))).any():
            raise ValueError(f"{column} must be numeric, not boolean.")
        result[column] = pd.to_numeric(result[column], errors="coerce")
        if not np.isfinite(result[column]).all() or not result[column].between(low, high).all():
            raise ValueError(f"{column} must be a finite number between {low} and {high}.")
    for column in CATEGORICAL + ([TARGET] if training else []):
        if not result[column].map(lambda value: isinstance(value, str) and bool(value.strip())).all():
            raise ValueError(f"{column} must contain nonempty text.")
        result[column] = result[column].str.strip()
    return result


def input_frame(values: Mapping[str, Any]) -> pd.DataFrame:
    """Validate one recommendation request."""
    return validate_frame(pd.DataFrame([dict(values)]))


def build_preprocessor() -> ColumnTransformer:
    """Encode categories; tree models do not require numerical scaling."""
    return ColumnTransformer([
        ("numeric", "passthrough", NUMERIC),
        ("categorical", OneHotEncoder(handle_unknown="ignore"), CATEGORICAL),
    ])
