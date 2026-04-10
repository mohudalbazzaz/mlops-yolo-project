from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from tensorflow.keras import layers, models
import numpy as np
import mlflow


def split_data(image_classifications):
    """
    Splits image data into training (70%), validation (15%), and test sets (15%), and encodes labels.

    Args:
        image_classifications (dict): Dictionary mapping class labels (str)
            to lists of images. Each image is expected to be a NumPy array
            of shape (height, width, channels).

    Returns:
        X_train (np.ndarray): Training images.
        y_train (np.ndarray): Encoded training labels.
        X_val (np.ndarray): Validation images.
        y_val (np.ndarray): Encoded validation labels.
        X_test (np.ndarray): Test images.
        y_test (np.ndarray): Encoded test labels.
    """
    X = []
    y = []

    for classification, images in image_classifications.items():
        # the images being extended to the lists are represented as arrays of pixel itensities i.e
        # [
        # [[0.502, 0.251, 0.0], [0.600, 0.300, 0.0], ..., [pixel128]],
        # [[0.450, 0.200, 0.1], [0.550, 0.250, 0.0], ..., [pixel128]],
        # ...  # 128 rows
        # ], ...
        X.extend(images)
        y.extend(len(images) * [classification])

    X = np.array(X)  # returns shape (280 (num images), 128, 128, 3)

    # train = 0.7, val = 0.15, test = 0.15
    X_train, X_temp, y_train, y_temp = train_test_split(
        X, y, test_size=0.3, stratify=y, random_state=42
    )

    X_val, X_test, y_val, y_test = train_test_split(
        X_temp, y_temp, test_size=0.5, stratify=y_temp, random_state=42
    )

    le = LabelEncoder()
    y_train = le.fit_transform(
        y_train
    )  # learns the mapping e.g Green --> 1 and applies it immediately. le instance stores mapping for val and test
    y_val = le.transform(y_val)
    y_test = le.transform(y_test)

    return X_train, y_train, X_val, y_val, X_test, y_test


def train_and_validate_model(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
    X_test: np.ndarray,
    y_test: np.ndarray,
    batch_size: int = 32,
) -> tuple:
    """
    Trains a CNN model for banana ripeness classification, validates it,
    evaluates on a test set, and logs experiment details to MLflow.

    Args:
        X_train (np.ndarray): Training images with shape
            (num_samples, height, width, channels).
        y_train (np.ndarray): Training labels.
        X_val (np.ndarray): Validation images.
        y_val (np.ndarray): Validation labels.
        X_test (np.ndarray): Test images.
        y_test (np.ndarray): Test labels.
        batch_size (int, optional): Number of samples per gradient update.
            Defaults to 32.

    Returns:
        History: Keras training history containing loss and accuracy per epoch.
        keras.Model: Trained CNN model.
        float: Accuracy on the held-out test set.
    """
    with mlflow.start_run():

        model = models.Sequential(
            [
                layers.Input(shape=X_train.shape[1:]),  # excluding the number of images
                layers.Conv2D(32, (3, 3), activation="relu"),
                layers.MaxPooling2D((2, 2)),
                layers.Conv2D(64, (3, 3), activation="relu"),
                layers.MaxPooling2D((2, 2)),
                layers.Flatten(),
                layers.Dense(128, activation="relu"),
                layers.Dense(3, activation="softmax"),
            ]
        )

        model.compile(
            optimizer="adam",
            loss="sparse_categorical_crossentropy",
            metrics=["accuracy"],
        )

        mlflow.log_param("batch_size", batch_size)
        mlflow.log_param("epochs", 8)
        mlflow.log_param("optimizer", "adam")

        # in batches backprop to update weights, forward prop using those weights to calculate loss for each epoch
        metrics = model.fit(
            X_train,
            y_train,
            validation_data=(X_val, y_val),
            epochs=8,
            batch_size=batch_size,
        )

        _, test_accuracy = model.evaluate(X_test, y_test)

        mlflow.log_metric("test_accuracy", test_accuracy)

        mlflow.tensorflow.log_model(model, "model")

        return metrics, model, test_accuracy
