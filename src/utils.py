"""Project paths and safe image decoding."""
from io import BytesIO
from pathlib import Path
import warnings
from PIL import Image, ImageOps, UnidentifiedImageError

ROOT = Path(__file__).resolve().parents[1]
MODEL_DIR = ROOT / "models"
MAX_IMAGE_BYTES = 10 * 1024 * 1024
MAX_IMAGE_PIXELS = 20_000_000


def decode_image(content: bytes) -> Image.Image:
    """Decode only JPEG/PNG, with byte and pixel limits and EXIF orientation."""
    if not content or len(content) > MAX_IMAGE_BYTES:
        raise ValueError("Choose a nonempty JPEG or PNG image no larger than 10 MB.")
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("error", Image.DecompressionBombWarning)
            with Image.open(BytesIO(content)) as probe:
                if probe.format not in {"JPEG", "PNG"}:
                    raise ValueError("Only actual JPEG and PNG images are accepted.")
                if probe.width * probe.height > MAX_IMAGE_PIXELS:
                    raise ValueError("Image exceeds the 20 megapixel limit.")
                probe.verify()
            with Image.open(BytesIO(content)) as source:
                return ImageOps.exif_transpose(source).convert("RGB")
    except (UnidentifiedImageError, OSError, SyntaxError, Image.DecompressionBombError,
            Image.DecompressionBombWarning) as exc:
        raise ValueError("The image is invalid, damaged, or too large to decode safely.") from exc


def model_version(path: Path) -> tuple[int, int]:
    """Cache key changes when a model artifact is replaced."""
    info = path.stat()
    return info.st_mtime_ns, info.st_size
