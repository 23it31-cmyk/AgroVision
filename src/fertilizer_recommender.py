"""Random Forest construction, persistence and recommendation helpers."""
from pathlib import Path
from typing import Any, Mapping
import joblib
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline
from src.preprocessing import CATEGORICAL, FEATURES, build_preprocessor, input_frame


def build_pipeline(random_state: int = 42) -> Pipeline:
    """Build a complete preprocessing and classification pipeline."""
    return Pipeline([
        ("preprocessor", build_preprocessor()),
        ("classifier", RandomForestClassifier(n_estimators=200, random_state=random_state,
                                               class_weight="balanced", n_jobs=1)),
    ])


def load_model(path: Path) -> Pipeline:
    """Load a trusted local joblib pipeline. Never accept uploaded model files."""
    if not path.is_file():
        raise FileNotFoundError("Train the fertilizer model before requesting predictions.")
    model = joblib.load(path)
    if not isinstance(model, Pipeline) or list(model.feature_names_in_) != FEATURES:
        raise ValueError("The model does not match the AgroVision feature schema.")
    if not isinstance(model.named_steps.get("classifier"), RandomForestClassifier):
        raise ValueError("Expected a fitted Random Forest pipeline.")
    return model


def supported_categories(model: Pipeline) -> dict[str, list[str]]:
    """Read fitted crop and soil categories for consistent UI choices."""
    encoder = model.named_steps["preprocessor"].named_transformers_["categorical"]
    return {key: list(map(str, values)) for key, values in zip(CATEGORICAL, encoder.categories_)}


def recommend(model: Pipeline, values: Mapping[str, Any]) -> dict[str, Any]:
    """Predict only supported categories and include the model vote probability."""
    frame = input_frame(values)
    for column, choices in supported_categories(model).items():
        if frame.iloc[0][column] not in choices:
            raise ValueError(f"Unsupported {column}; select a category represented in training data.")
    label = str(model.predict(frame)[0])
    confidence = None
    if hasattr(model, "predict_proba"):
        probabilities = model.predict_proba(frame)[0]
        confidence = float(probabilities[list(model.classes_).index(label)])
    return {"fertilizer": label, "confidence": confidence}
