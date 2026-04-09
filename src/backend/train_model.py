import tensorflow as tf
import os

from src.backend.model_utils import split_data, train_and_validate_model
from src.backend.supabase import extract_imgs_from_db

main_bucket = os.environ.get("MAIN_BUCKET")

MODEL_NAME = "banana_model.keras"

def save_model() -> None:
    """Trains a model on stored images and save it to disk."""
    images = extract_imgs_from_db(main_bucket)

    X_train, y_train, X_val, y_val, X_test, y_test = split_data(images)

    _, model, _ = train_and_validate_model(X_train, y_train, X_val, y_val, X_test, y_test)

    model.save(MODEL_NAME)

def load_model():
    return tf.keras.models.load_model(MODEL_NAME)

if __name__ == "__main__":
    save_model()