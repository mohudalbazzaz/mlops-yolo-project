import io
import numpy as np
from PIL import Image

from src.backend.general_utils import preprocess_image


def test_preprocess_image_output_shape_and_range():
    img = Image.fromarray(np.random.randint(0, 256, (200, 200, 3), dtype=np.uint8))

    img_bytes = io.BytesIO()
    img.save(img_bytes, format="JPEG")
    file_bytes = img_bytes.getvalue()

    processed = preprocess_image(file_bytes)

    assert isinstance(processed, np.ndarray)
    assert processed.shape == (128, 128, 3)
    assert processed.min() >= 0.0
    assert processed.max() <= 1.0
