import numpy as np
import mlflow

from src.backend.train_model import load_model
from src.backend.blob_storage import extract_imgs_from_db

model = load_model()

BANANA_PRODUCTION_IMAGES = "banana-production-images"


def monitor_model() -> None:
    """
    Loads labelled test images from a storage bucket, performs inference
    using a pre-loaded model, and logs each prediction to MLflow.
    """
    image_classifications = extract_imgs_from_db(BANANA_PRODUCTION_IMAGES)

    X = []
    y = []

    for classification, images in image_classifications.items():

        X.extend(images)
        y.extend(len(images) * [classification])

    for img, true_label in zip(X, y):

        img = np.expand_dims(img, axis=0)

        prediction_probs = model.predict(img)

        class_names = ["overripe", "ripe", "unripe"]

        predicted_class = class_names[np.argmax(prediction_probs)]

        mlflow.set_experiment("Banana monitoring")

        with mlflow.start_run(nested=True):
            mlflow.log_param("true_label", true_label)
            mlflow.log_param("predicted_class", predicted_class)
            mlflow.log_metric("correct", int(predicted_class == true_label))


if __name__ == "__name__":
    monitor_model()
