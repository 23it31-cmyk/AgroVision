# Adaptive AgroVision Framework for Precision Pest Detection and Intelligent Fertilizer Recommendation

A final-year academic Streamlit application combining custom YOLOv8 pest detection with a Random Forest fertilizer classifier. The dashboard works without model files and explains how to enable each workflow. **The bundled fertilizer data is synthetic demonstration data. No trained pest detector is supplied.**

## Problem statement

Crop inspection and interpretation of soil measurements require consistent observation and specialist knowledge. This project demonstrates how image detection and structured machine learning can organize those inputs into a simple decision-support interface. It does not establish improved yields or replace field diagnosis.

## Objectives

- Detect and annotate pests with a custom-trained YOLOv8 detector.
- Classify fertilizer labels from nine soil, crop and environmental inputs.
- Present understandable results with confidence estimates and clear limitations.
- Separate interface, validation, inference and training for reproducible development.

## System architecture

```text
User
  |
  v
Streamlit Web Application
  |
  +---------------------------+
  |                           |
  v                           v
Pest Detection          Fertilizer Recommendation
  |                           |
YOLOv8                  Input preprocessing
  |                           |
Pest + confidence       Random Forest
  |                           |
  +------------+--------------+
               |
               v
        Recommendation UI
```

“Adaptive” means models can be retrained and replaced; automatic or online adaptation is not implemented. Replacing a model file invalidates its resource cache using file modification time and size.

## Features

- Green dashboard with Home, Pest Detection, Fertilizer Recommendation and About Project pages.
- JPEG/PNG preview, safe decoding, 10 MB upload limit and 20 megapixel decoded-image limit.
- Custom weights only: annotated bounding boxes, classes, confidence and detection count.
- Validated numeric inputs and crop/soil selectors derived from the fitted encoder.
- Persisted `ColumnTransformer` + `OneHotEncoder` + `RandomForestClassifier` pipeline.
- Honest missing-model, invalid-input and model-load failure states.
- Synthetic-data provenance shown alongside demo predictions.

## Technology stack

Python 3.11+ (3.11 or 3.12 recommended for dependency availability), Streamlit, Ultralytics YOLOv8, scikit-learn, Pandas, NumPy, Pillow, OpenCV and Joblib. Plotly is included for optional extensions; the current dashboard uses native Streamlit metrics and tables. Numerical features pass through unchanged because Random Forest trees do not need `StandardScaler`.

## Folder structure

```text
agrovision-final-project/
├── app.py
├── requirements.txt
├── README.md
├── .gitignore
├── .streamlit/config.toml
├── pytest.ini
├── src/
│   ├── __init__.py
│   ├── pest_detector.py
│   ├── fertilizer_recommender.py
│   ├── preprocessing.py
│   └── utils.py
├── training/
│   ├── __init__.py
│   ├── train_yolo.py
│   └── train_fertilizer.py
├── models/README.md
├── data/
│   ├── fertilizer_sample.csv
│   └── README.md
├── assets/README.md
└── tests/
    ├── test_fertilizer.py
    ├── test_utils.py
    └── test_app.py
```

## Installation

From this project directory:

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu
python -m pip install -r requirements.txt
```

Use the installed `python3` command on Linux; a separate `python3.11` executable is not required. The command above installs CPU-only PyTorch for the demo, following the [official PyTorch installation guidance](https://docs.pytorch.org/get-started/locally/).

On Windows use `py -3.11 -m venv .venv` and `.venv\Scripts\Activate.ps1` in PowerShell. Dependencies use bounded compatible API ranges rather than an exact cross-platform lockfile. Freeze your validated environment for reproducible research. GPU training requires a PyTorch build appropriate to your hardware; CPU is the default here. Ultralytics may also bring its OpenCV dependency; no desktop OpenCV windows are used. If PyTorch has no wheel for your Python/platform, use Python 3.11/3.12.

## Fertilizer dataset requirements

Expected CSV header:

```csv
Nitrogen,Phosphorus,Potassium,Temperature,Humidity,Moisture,pH,Crop,Soil_Type,Fertilizer
```

| Columns | Demo units | Accepted bounds |
|---|---|---|
| Nitrogen, Phosphorus, Potassium | mg/kg | 0–500 each |
| Temperature | °C | −10–60 |
| Humidity, Moisture | % | 0–100 each |
| pH | dimensionless | 0–14 |
| Crop, Soil_Type, Fertilizer | nonempty text | training categories |

These are software bounds, not agronomic thresholds. Real training and inference must share measurement units and protocols. Missing/nonfinite values and duplicate feature rows are rejected. At least two classes and two rows per class are needed, with enough train/test rows for stratification. Small datasets do not support reliable performance conclusions.

The sample has 96 artificial records and four fertilizer labels. Read [data provenance](data/README.md) before using it. Use appropriately licensed, expert-labeled real observations for research and separate farms/sites/seasons during evaluation. The simple random split is only a demonstration baseline. Unsupported crop/soil categories are rejected at prediction time even though the encoder can technically handle them.

## YOLO dataset format

Supply a labeled object detection dataset with matching image/label stems:

```text
pests/
├── images/train/leaf001.jpg
├── images/val/leaf002.jpg
├── labels/train/leaf001.txt
├── labels/val/leaf002.txt
└── pests.yaml
```

Example `pests.yaml` (resolve `path` to your dataset location):

```yaml
path: /absolute/path/to/pests
train: images/train
val: images/val
names:
  0: aphid
  1: whitefly
```

Each label line is `class_id x_center y_center width height`, with box coordinates normalized to 0–1. For example, `0 0.50 0.45 0.20 0.10`. Class IDs are zero-based. Supply separate validation images and maintain a further independent test set for credible evaluation. The class names above are illustrative; the application displays the names embedded in your actual checkpoint.

## Train YOLOv8

```bash
python training/train_yolo.py --data path/to/pests.yaml --model yolov8n.pt --epochs 50 --imgsz 640 --batch 16 --device cpu
```

Training may download the starting YOLOv8 weights and requires network access on the first run. A generic pretrained checkpoint is only an initialization for training, never a pest detector by itself. The script trains on your labels and copies the best checkpoint to `models/pest_yolov8.pt`. `--device 0` selects a supported CUDA GPU. `--output` changes the destination; the application expects the default path. Training logs remain under `runs/`.

Alternatively, copy your own trusted pest-trained detection checkpoint to that path. Verify its dataset provenance and class mapping: the filename alone cannot prove that it was trained on pests. Without weights, the UI displays training instructions and still previews images. Inference never falls back to generic weights.

## Train Random Forest

```bash
python training/train_fertilizer.py --data data/fertilizer_sample.csv
```

Optional arguments: `--output models/fertilizer_rf.joblib --test-size 0.25 --seed 42`. Use `--synthetic` when supplying another synthetic dataset. The bundled sample is marked automatically.

Training validates the CSV, makes a reproducible stratified split, fits preprocessing only on the training partition, evaluates the held-out partition, and saves the complete pipeline. Actual accuracy, classification report, label order and confusion matrix are printed and written to `models/fertilizer_rf_metrics.json`. These computed scores on synthetic data are **not research results**. The saved model is the model evaluated on the holdout; it is not silently refitted on that holdout.

## Run Streamlit

```bash
streamlit run app.py
```

Run from this project directory so `.streamlit/config.toml` is loaded. The browser normally opens at `http://localhost:8501`. Both model paths are anchored to the project location rather than the shell working directory. Load only trusted local model files; joblib and PyTorch deserialization can execute code. This app never accepts uploaded models.

## Example demonstration workflow

1. Start the app and show the Home workflow and readiness indicators.
2. Open Fertilizer Recommendation before training to demonstrate the missing-model message.
3. Train on the sample CSV, then return to the page. Explain the synthetic-demo warning.
4. Enter compatible measurements, select a crop and soil, and submit. Review the fertilizer label, N/P/K and confidence estimate.
5. Open Pest Detection and upload a small JPEG/PNG. Without custom weights, preview the image and show the setup message.
6. After training a real pest detector, upload an independently labeled image and inspect boxes/classes. A zero-detection result does not prove that a plant is healthy.

## Verification

```bash
python -m compileall -q app.py src training tests
python -m pytest -q
python training/train_fertilizer.py --help
python training/train_yolo.py --help
```

Tests cover validation, nonfinite values, unsupported categories, persistence, training metrics structure, corrupt and oversized images, missing models, Streamlit navigation and demo prediction. Tests train temporary demo pipelines and do not require YOLO weights. Actual pest inference and full YOLO training require Ultralytics and your real dataset/checkpoint.

### Verification performed in this workspace

Verified in the project's `.venv`: Python 3.14.7, Streamlit 1.64.0, scikit-learn 1.9.1, Pandas 3.0.6, NumPy 2.5.2, PyTorch 2.14.0+cpu and Ultralytics 8.4.160. All requirements were installed, `pip check` passed, and all 24 pytest tests passed. Ultralytics, torchvision and OpenCV imports succeeded. The synthetic fertilizer model was retrained in this environment, and Streamlit's health endpoint returned `ok`. Actual pest training/inference still requires a real labeled dataset or trusted pest-trained checkpoint.

## Limitations

- No supplied pest model or pest image dataset, and no demonstrated pest accuracy.
- Synthetic fertilizer data lacks field validity; classification labels are not fertilizer dosage guidance.
- Model confidence is not calibrated probability of agronomic correctness.
- Limited training categories and no robust out-of-distribution detector.
- No site/season-aware cross-validation, external validation, online learning or yield study.
- Shared YOLO inference is serialized to protect the cached model; large-scale serving is outside scope.
- User-facing model failures show concise messages; server logs retain diagnostics.

## Future enhancements

Collect region-specific, expert-reviewed data; implement site-separated evaluation and probability calibration; validate pest detection mAP/precision/recall on an independent test set; integrate soil sensors; add language support, drift monitoring and agronomist-reviewed dose calculations.

## Academic disclaimer

This is a software prototype for education and demonstration. Synthetic data and scores must not be presented as real experimental evidence. Do not claim field performance, yield improvements or agronomic effectiveness without a valid study. Recommendations are decision-support only and must be checked against local agronomic guidance and laboratory soil tests. No pesticide or fertilizer application rates are prescribed.

## API references

- [Ultralytics training documentation](https://docs.ultralytics.com/modes/train/)
- [Streamlit file uploader](https://docs.streamlit.io/develop/api-reference/widgets/st.file_uploader)
- [scikit-learn OneHotEncoder](https://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.OneHotEncoder.html)
