"""Exercise validation, preprocessing, training and persisted prediction."""
import numpy as np
import pandas as pd
import pytest
from src.fertilizer_recommender import build_pipeline, load_model, recommend
from src.preprocessing import FEATURES, input_frame, validate_frame
from src.utils import ROOT
from training.train_fertilizer import train


@pytest.fixture
def frame():
    return pd.read_csv(ROOT / 'data/fertilizer_sample.csv')


@pytest.mark.parametrize('column,value', [('Nitrogen', -1), ('pH', 15), ('Moisture', np.nan),
                                         ('Temperature', np.inf), ('Humidity', 'bad'), ('Crop', ''),
                                         ('Soil_Type', None), ('Potassium', True)])
def test_invalid_inputs(frame, column, value):
    values = frame.iloc[0][FEATURES].to_dict()
    values[column] = value
    with pytest.raises(ValueError):
        input_frame(values)


def test_schema_and_empty(frame):
    with pytest.raises(ValueError, match='Missing'):
        validate_frame(frame.drop(columns='pH'), training=True)
    with pytest.raises(ValueError, match='no rows'):
        validate_frame(frame.iloc[:0])


def test_pipeline_and_unknown_categories(frame):
    model = build_pipeline().fit(frame[FEATURES], frame.Fertilizer)
    values = frame.iloc[0][FEATURES].to_dict()
    result = recommend(model, values)
    assert result['fertilizer'] in set(frame.Fertilizer)
    assert 0 <= result['confidence'] <= 1
    values['Crop'] = 'Unseen crop'
    # Encoder is robust, but serving rejects unsupported agronomic categories.
    assert model.named_steps['preprocessor'].transform(input_frame(values)).shape[0] == 1
    with pytest.raises(ValueError, match='Unsupported'):
        recommend(model, values)


def test_training_round_trip(tmp_path, frame):
    output = tmp_path / 'fertilizer.joblib'
    metrics = train(ROOT / 'data/fertilizer_sample.csv', output)
    model = load_model(output)
    assert metrics['synthetic_demo'] is True
    assert metrics['train_rows'] + metrics['test_rows'] == len(frame)
    assert np.asarray(metrics['confusion_matrix']).sum() == metrics['test_rows']
    assert (tmp_path / 'fertilizer_metrics.json').exists()
    assert recommend(model, frame.iloc[0][FEATURES].to_dict())['fertilizer'] in model.classes_


def test_bad_training_data(tmp_path, frame):
    path = tmp_path / 'bad.csv'
    pd.concat([frame, frame.iloc[:1]]).to_csv(path, index=False)
    with pytest.raises(ValueError, match='Duplicate'):
        train(path, tmp_path / 'bad.joblib')
    frame.iloc[:2].to_csv(path, index=False)
    with pytest.raises(ValueError, match='two classes'):
        train(path, tmp_path / 'bad.joblib')


def test_missing_model(tmp_path):
    with pytest.raises(FileNotFoundError):
        load_model(tmp_path / 'missing.joblib')
