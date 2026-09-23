# Model artifacts

No trained pest weights are bundled. Put a trusted, pest-specific YOLOv8 detection checkpoint at `pest_yolov8.pt`. A generic COCO checkpoint is not a pest detector; renaming it does not make it one. Check class names and dataset provenance yourself.

`python training/train_fertilizer.py` creates `fertilizer_rf.joblib` and `fertilizer_rf_metrics.json`. The pipeline includes preprocessing and records synthetic-demo provenance. Metrics describe the held-out split only. Model binaries and generated metrics are ignored by Git.

Only load trusted local checkpoints: pickle/joblib and some PyTorch artifacts may execute code when loaded. Uploaded model files are not supported. Retrain when changing dependency versions; joblib portability across scikit-learn versions is not guaranteed.
