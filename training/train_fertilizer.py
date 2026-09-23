"""Train and evaluate a fertilizer pipeline; synthetic scores are demo-only."""
import argparse
import json
import math
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import joblib
import pandas as pd
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.model_selection import train_test_split
from src.fertilizer_recommender import build_pipeline
from src.preprocessing import FEATURES, TARGET, validate_frame
from src.utils import ROOT


def train(data: Path, output: Path, test_size: float = 0.25, seed: int = 42,
          synthetic: bool = False) -> dict:
    """Validate, split, fit, evaluate and persist pipeline plus provenance."""
    if not 0 < test_size < 1:
        raise ValueError("test-size must be between 0 and 1.")
    frame = validate_frame(pd.read_csv(data), training=True)
    if frame[FEATURES].duplicated().any():
        raise ValueError("Duplicate feature rows found; deduplicate before splitting to avoid leakage.")
    counts = frame[TARGET].value_counts()
    n_test = math.ceil(len(frame) * test_size)
    if len(counts) < 2 or counts.min() < 2 or min(n_test, len(frame) - n_test) < len(counts):
        raise ValueError("Need at least two classes, two rows per class, and enough train/test rows for all classes.")
    x_train, x_test, y_train, y_test = train_test_split(
        frame[FEATURES], frame[TARGET], test_size=test_size, random_state=seed, stratify=frame[TARGET])
    model = build_pipeline(seed)
    model.fit(x_train, y_train)
    predictions = model.predict(x_test)
    labels = list(map(str, model.classes_))
    is_synthetic = synthetic or data.resolve() == (ROOT / "data/fertilizer_sample.csv").resolve()
    provenance = {"synthetic_demo": is_synthetic, "dataset": data.name,
                  "train_rows": len(x_train), "test_rows": len(x_test), "seed": seed}
    model.agrovision_metadata_ = provenance
    metrics = {**provenance, "accuracy": float(accuracy_score(y_test, predictions)),
               "labels": labels, "classification_report": classification_report(
                   y_test, predictions, labels=labels, output_dict=True, zero_division=0),
               "confusion_matrix": confusion_matrix(y_test, predictions, labels=labels).tolist()}
    output.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, output)
    output.with_name(output.stem + "_metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    return metrics


def main() -> None:
    """Command-line training entry point."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=ROOT / "data/fertilizer_sample.csv")
    parser.add_argument("--output", type=Path, default=ROOT / "models/fertilizer_rf.joblib")
    parser.add_argument("--test-size", type=float, default=0.25)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--synthetic", action="store_true", help="Mark other synthetic datasets as demo data")
    args = parser.parse_args()
    try:
        metrics = train(args.data, args.output, args.test_size, args.seed, args.synthetic)
    except (ValueError, OSError) as exc:
        parser.exit(1, f"Training failed: {exc}\n")
    print("SYNTHETIC DEMO ONLY — not research evidence." if metrics["synthetic_demo"]
          else "Held-out split results; independent field validation is still required.")
    print(json.dumps(metrics, indent=2))
    print(f"Saved pipeline: {args.output}")


if __name__ == "__main__":
    main()
