"""Validate image payloads and model cache invalidation."""
from io import BytesIO
from PIL import Image
import pytest
from src import utils
from src.pest_detector import PestDetector


def encoded(format='PNG'):
    buffer = BytesIO()
    Image.new('RGB', (12, 8), 'green').save(buffer, format=format)
    return buffer.getvalue()


@pytest.mark.parametrize('format', ['PNG', 'JPEG'])
def test_image(format):
    image = utils.decode_image(encoded(format))
    assert image.size == (12, 8)
    assert image.mode == 'RGB'


@pytest.mark.parametrize('content', [b'', b'not an image', encoded('GIF'), encoded()[:25]])
def test_invalid_images(content):
    with pytest.raises(ValueError):
        utils.decode_image(content)


def test_limits(monkeypatch):
    monkeypatch.setattr(utils, 'MAX_IMAGE_BYTES', 5)
    with pytest.raises(ValueError, match='10 MB'):
        utils.decode_image(encoded())
    monkeypatch.setattr(utils, 'MAX_IMAGE_BYTES', 10000)
    monkeypatch.setattr(utils, 'MAX_IMAGE_PIXELS', 10)
    with pytest.raises(ValueError, match='megapixel'):
        utils.decode_image(encoded())


def test_version(tmp_path):
    path = tmp_path / 'model'
    path.write_bytes(b'a')
    before = utils.model_version(path)
    path.write_bytes(b'updated')
    assert utils.model_version(path) != before


def test_missing_pest_weights(tmp_path):
    with pytest.raises(FileNotFoundError, match='Custom pest'):
        PestDetector(tmp_path / 'missing.pt')
