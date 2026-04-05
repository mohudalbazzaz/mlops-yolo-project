import numpy as np
from tensorflow.keras import models

from src.model_utils import split_data, train_and_validate_model

def test_split_data_shapes_and_labels():
    fake_images = {
        "Ripe": [np.random.rand(128, 128, 3) for _ in range(10)],
        "Unripe": [np.random.rand(128, 128, 3) for _ in range(10)]
    }

    X_train, y_train, X_val, y_val, X_test, y_test = split_data(fake_images)

    assert X_train.shape[0] + X_val.shape[0] + X_test.shape[0] == 20
    assert X_train.shape[1:] == (128, 128, 3)
    assert X_val.shape[1:] == (128, 128, 3)
    assert X_test.shape[1:] == (128, 128, 3)

    assert set(y_train).issubset({0, 1})
    assert set(y_val).issubset({0, 1})
    assert set(y_test).issubset({0, 1})

    # 0.7 of 20 is 14, 0.15 of 20 is 3
    assert len(y_train) == 14
    assert len(y_val) == 3
    assert len(y_test) == 3


def test_train_and_validate_model():
    X_train = np.random.rand(4, 128, 128, 3)
    y_train = np.array([0, 1, 2, 0])
    X_val = np.random.rand(2, 128, 128, 3)
    y_val = np.array([1, 2])
    X_test = np.random.rand(2, 128, 128, 3)
    y_test = np.array([0, 2])

    metrics, model = train_and_validate_model(X_train, y_train, X_val, y_val, X_test, y_test, batch_size=2)

    assert isinstance(model, models.Sequential)
    assert metrics.history['loss']
    assert metrics.history['accuracy']
    assert model.output_shape[-1] == 3