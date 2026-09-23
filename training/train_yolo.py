"""Train YOLOv8 on a user-supplied, pest-labeled detection dataset."""
import argparse
from pathlib import Path
import shutil
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.utils import ROOT


def main() -> None:
    """Configure training and copy best weights to the application model path."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, required=True, help="YOLO detection dataset YAML")
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--model", default="yolov8n.pt")
    parser.add_argument("--batch", type=int, default=16)
    parser.add_argument("--device", default="cpu", help="cpu, 0 for a CUDA GPU, or mps")
    parser.add_argument("--output", type=Path, default=ROOT / "models/pest_yolov8.pt")
    args = parser.parse_args()
    if not args.data.is_file() or args.data.suffix.lower() not in {".yaml", ".yml"}:
        parser.error("--data must point to an existing dataset YAML.")
    if min(args.epochs, args.imgsz, args.batch) <= 0:
        parser.error("epochs, imgsz and batch must be positive.")
    try:
        from ultralytics import YOLO
        model = YOLO(args.model, task="detect")
        model.train(data=str(args.data.resolve()), epochs=args.epochs, imgsz=args.imgsz,
                    batch=args.batch, device=args.device, project=str(ROOT / "runs"), name="pest", seed=42)
        best = Path(model.trainer.best)
        if not best.is_file():
            raise RuntimeError("Training did not produce a best checkpoint.")
        args.output.parent.mkdir(parents=True, exist_ok=True)
        if best.resolve() != args.output.resolve():
            shutil.copy2(best, args.output)
    except Exception as exc:
        parser.exit(1, f"YOLO training failed: {exc}\n")
    print(f"Saved custom checkpoint to {args.output}. Validate it on independent pest images before use.")


if __name__ == "__main__":
    main()
