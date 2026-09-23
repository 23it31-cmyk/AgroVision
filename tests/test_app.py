"""Smoke-test navigation and prediction using Streamlit's application harness."""
from streamlit.testing.v1 import AppTest
from src import utils
from training.train_fertilizer import train


def test_navigation_without_models(monkeypatch, tmp_path):
    monkeypatch.setattr(utils, 'MODEL_DIR', tmp_path)
    app = AppTest.from_file(str(utils.ROOT / 'app.py')).run()
    assert not app.exception
    for page in ['Pest Detection', 'Fertilizer Recommendation', 'About Project', 'Home']:
        app.sidebar.radio[0].set_value(page).run()
        assert not app.exception
        if page in ['Pest Detection', 'Fertilizer Recommendation']:
            assert app.warning


def test_fertilizer_prediction(monkeypatch, tmp_path):
    train(utils.ROOT / 'data/fertilizer_sample.csv', tmp_path / 'fertilizer_rf.joblib')
    monkeypatch.setattr(utils, 'MODEL_DIR', tmp_path)
    app = AppTest.from_file(str(utils.ROOT / 'app.py')).run()
    app.sidebar.radio[0].set_value('Fertilizer Recommendation').run()
    assert not app.exception
    app.button[0].click().run()
    assert not app.exception
    assert any('Recommended fertilizer:' in item.value for item in app.subheader)
    assert any('SYNTHETIC DEMO' in item.value for item in app.warning)
