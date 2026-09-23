"""AgroVision: a transparent academic demonstration dashboard."""
import logging
from pathlib import Path
import pandas as pd
import streamlit as st
from src.fertilizer_recommender import build_demo_model, load_model, recommend, supported_categories
from src.pest_detector import PestDetector
from src.preprocessing import RANGES
from src.utils import ROOT, MODEL_DIR, decode_image, model_version

TITLE = "Adaptive AgroVision Framework for Precision Pest Detection and Intelligent Fertilizer Recommendation"
LOGGER = logging.getLogger(__name__)
st.set_page_config(page_title="AgroVision | Precision Agriculture", page_icon="🌿", layout="wide")


@st.cache_resource(show_spinner="Loading fertilizer pipeline…")
def cached_fertilizer(path: str, version: tuple[int, int]):
    """Cache trusted model artifacts, invalidating on file replacement."""
    return load_model(Path(path))


@st.cache_resource(show_spinner="Preparing synthetic demonstration model…")
def cached_demo(path: str, version: tuple[int, int]):
    """Build the optional demo once per server and source dataset version."""
    return build_demo_model(Path(path))


@st.cache_resource(show_spinner="Loading custom pest detector…")
def cached_detector(path: str, version: tuple[int, int]) -> PestDetector:
    """Cache the thread-safe YOLO wrapper across Streamlit reruns."""
    return PestDetector(Path(path))


def home() -> None:
    """Present the project purpose and transparent readiness status."""
    st.caption("PRECISION AGRICULTURE · FINAL-YEAR PROJECT")
    st.title("Better observations. Informed crop care.")
    st.subheader(TITLE)
    st.write("Explore pest detection from plant images and fertilizer classification from soil and crop inputs in one focused workspace.")
    cols = st.columns(3)
    cols[0].metric("Analysis workflows", "2")
    cols[1].metric("Pest model", "Available" if (MODEL_DIR / "pest_yolov8.pt").is_file() else "Awaiting weights")
    cols[2].metric("Fertilizer model", "Available" if (MODEL_DIR / "fertilizer_rf.joblib").is_file() else "Awaiting training")
    left, right = st.columns(2)
    with left.container(border=True):
        st.subheader("🔍 Pest Detection")
        st.write("Upload a crop image and inspect pest classes, bounding boxes and confidence from your custom YOLOv8 detector.")
        st.caption("JPEG / PNG · maximum 10 MB · custom weights required")
    with right.container(border=True):
        st.subheader("🌱 Fertilizer Recommendation")
        st.write("Combine nutrient measurements, environment, crop and soil type with a Random Forest classification pipeline.")
        st.caption("9 input features · complete preprocessing pipeline")
    st.subheader("From observation to decision support")
    for col, title, detail in zip(st.columns(4), ["01 · Collect", "02 · Validate", "03 · Analyze", "04 · Review"],
                                ["Plant image or soil inputs →", "Image safety and input checks →", "YOLOv8 or Random Forest →", "Review results with an agronomist"]):
        with col.container(border=True):
            st.markdown(f"**{title}**")
            st.write(detail)
    st.info("Academic demonstration: the bundled fertilizer dataset is synthetic. No pest weights or field-validated recommendations are included.")
    st.caption("Python · Streamlit · Ultralytics YOLOv8 · scikit-learn · Pandas · NumPy · Pillow · OpenCV · Joblib")


def pests() -> None:
    """Upload, safely decode, and run optional custom pest inference."""
    st.title("Pest Detection")
    st.write("Inspect visible pests using a custom-trained YOLOv8 model.")
    path = MODEL_DIR / "pest_yolov8.pt"
    if not path.is_file():
        st.warning("Custom pest model required. Train YOLOv8 on a labeled pest dataset or copy trusted, pest-trained weights to models/pest_yolov8.pt. Image preview remains available.")
        st.code("python training/train_yolo.py --data path/to/pests.yaml --model yolov8n.pt --epochs 50 --imgsz 640")
    uploaded = st.file_uploader("Choose a plant or leaf image", type=["jpg", "jpeg", "png"])
    threshold = st.slider("Detection confidence threshold", 0.05, 0.95, 0.25, 0.05)
    st.caption("Maximum 10 MB and 20 megapixels. Confidence is a model score, not a guarantee of correct identification.")
    if uploaded is None:
        return
    try:
        image = decode_image(uploaded.getvalue())
    except ValueError as exc:
        st.error(str(exc))
        return
    left, right = st.columns(2)
    left.image(image, caption="Uploaded image", use_container_width=True)
    if st.button("Detect pests", type="primary", disabled=not path.is_file()):
        try:
            detector = cached_detector(str(path), model_version(path))
            with st.spinner("Analyzing image…"):
                annotated, records = detector.detect(image, threshold)
            right.image(annotated, caption="Detection results", use_container_width=True)
            st.metric("Number of detections", len(records))
            if records:
                st.dataframe(pd.DataFrame(records), hide_index=True, use_container_width=True)
            else:
                st.info("No pests detected at this threshold. This does not establish that the plant is pest-free.")
        except Exception:
            LOGGER.exception("Pest inference failed")
            st.error("Pest detection could not run. Check the installed dependencies and that the checkpoint is a compatible, trusted pest detection model.")


def fertilizer() -> None:
    """Collect validated features and display a fertilizer prediction."""
    st.title("Fertilizer Recommendation")
    st.write("Enter soil and environmental measurements to explore the trained classifier.")
    st.caption("N, P, K: mg/kg · Temperature: °C · Humidity and moisture: % · pH: 0–14. Use training-compatible measurement methods.")
    path = MODEL_DIR / "fertilizer_rf.joblib"
    if not path.is_file():
        st.warning("Fertilizer model not found. Train the pipeline to enable recommendations.")
        st.code("python training/train_fertilizer.py --data data/fertilizer_sample.csv")
        st.info("For a hosted demonstration, use the bundled synthetic data to prepare a temporary model. No terminal or uploaded model is needed. This is not experimental research data.")
        if st.button("Start synthetic demo", type="primary"):
            st.session_state["fertilizer_demo_enabled"] = True
        if not st.session_state.get("fertilizer_demo_enabled", False):
            return
    try:
        if path.is_file():
            model = cached_fertilizer(str(path), model_version(path))
        else:
            demo_path = ROOT / "data/fertilizer_sample.csv"
            model = cached_demo(str(demo_path), model_version(demo_path))
        categories = supported_categories(model)
    except Exception:
        LOGGER.exception("Fertilizer model loading failed")
        st.error("Unable to load the fertilizer pipeline. Retrain it in this environment using the documented command.")
        return
    metadata = getattr(model, "agrovision_metadata_", {})
    if metadata.get("synthetic_demo"):
        st.warning("SYNTHETIC DEMO MODEL — outputs illustrate software behavior only. They are not validated agronomic recommendations or research results.")
    else:
        st.info("Model provenance and local suitability must be verified before practical use.")
    defaults = dict(zip(RANGES, [25.0, 60.0, 65.0, 27.0, 65.0, 45.0, 6.5]))
    values = {}
    with st.form("fertilizer_inputs"):
        cols = st.columns(3)
        for i, (name, bounds) in enumerate(RANGES.items()):
            values[name] = cols[i % 3].number_input(name, min_value=bounds[0], max_value=bounds[1], value=defaults[name], step=0.1)
        values["Crop"] = cols[1].selectbox("Crop type", categories["Crop"])
        values["Soil_Type"] = cols[2].selectbox("Soil type", categories["Soil_Type"])
        submitted = st.form_submit_button("Recommend fertilizer", type="primary")
    if submitted:
        try:
            result = recommend(model, values)
        except ValueError as exc:
            st.error(str(exc))
            return
        except Exception:
            LOGGER.exception("Fertilizer prediction failed")
            st.error("Prediction failed. Check the model compatibility and retrain if necessary.")
            return
        with st.container(border=True):
            st.subheader(f"Recommended fertilizer: {result['fertilizer']}")
            st.write(f"Crop: {values['Crop']} · Soil: {values['Soil_Type']}")
            metrics = st.columns(4)
            for col, name in zip(metrics, ["Nitrogen", "Phosphorus", "Potassium"]):
                col.metric(name, f"{values[name]:g} mg/kg")
            if result["confidence"] is not None:
                metrics[3].metric("Prediction confidence", f"{result['confidence']:.1%}")
            st.caption("Confidence is the forest’s class probability estimate; it is not calibrated field reliability.")
    st.info("Decision-support only. Validate recommendations against local agronomic guidance, soil testing and crop requirements. This application does not calculate application rates.")


def about() -> None:
    """Explain architecture and academic limitations."""
    st.title("About Project")
    st.subheader(TITLE)
    st.write("The framework combines visual pest observations with structured soil analysis to demonstrate two complementary machine-learning workflows for precision agriculture.")
    st.code("User → Streamlit\n  ├─ Image → YOLOv8 → Pest classes + confidence + boxes\n  └─ Soil/crop inputs → ColumnTransformer → Random Forest\n                         ↓\n                Recommendation UI")
    st.markdown("**Objectives:** simplify inspection, standardize model input validation, and present interpretable prediction summaries in an accessible dashboard.")
    st.markdown("**Limitations:** no supplied pest checkpoint; synthetic fertilizer data; limited categories; uncalibrated confidence; no dosage recommendation; no field validation or automatic learning.")
    st.write("‘Adaptive’ describes the ability to retrain and replace models. Online adaptation, sensor integration, regional calibration and independent field evaluation are future work.")
    st.warning("Academic integrity: synthetic records and their evaluation scores must never be presented as real experimental research results.")


with st.sidebar:
    st.title("🌿 AgroVision")
    st.caption("PRECISION AGRICULTURE")
    page = st.radio("Navigation", ["Home", "Pest Detection", "Fertilizer Recommendation", "About Project"])
    st.divider()
    st.caption("Observe · Analyze · Review")
    st.caption("Academic prototype / decision support")
{"Home": home, "Pest Detection": pests, "Fertilizer Recommendation": fertilizer, "About Project": about}[page]()
st.divider()
st.caption("AgroVision · Transparent models. Responsible interpretation.")
