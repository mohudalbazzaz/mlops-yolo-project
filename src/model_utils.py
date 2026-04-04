import tensorflow as tf
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from tensorflow.keras import layers, models
import numpy as np
import matplotlib.pyplot as plt

def split_data(image_classifications):

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

    X = np.array(X) # returns shape (280, 128, 128, 3)

    # train = 0.7, val = 0.15, test = 0.15
    X_train, X_temp, y_train, y_temp = train_test_split(
        X, y, test_size=0.3, stratify=y, random_state=42
    )

    X_val, X_test, y_val, y_test = train_test_split(
        X_temp, y_temp, test_size=0.5, stratify=y_temp, random_state=42
    )

    le = LabelEncoder()
    y_train = le.fit_transform(y_train) # learns the mapping e.g Green --> 1 and applies it immediately. le instance stores mapping for val and test
    y_val = le.transform(y_val)
    y_test = le.transform(y_test)

    return X_train, y_train, X_val, y_val, X_test, y_test

def train_and_validate_model(X_train, y_train, X_val, y_val, X_test, y_test, batch_size=32):

    model = models.Sequential([
    layers.Input(shape=X_train.shape[1:]), # excluding the number of images 
    layers.Conv2D(32, (3,3), activation='relu'),
    layers.MaxPooling2D((2,2)),
    layers.Conv2D(64, (3,3), activation='relu'),
    layers.MaxPooling2D((2,2)),
    layers.Flatten(),
    layers.Dense(128, activation='relu'),
    layers.Dense(3, activation='softmax')
    ])
    
    model.compile(optimizer='adam',
                loss='sparse_categorical_crossentropy',
                metrics=['accuracy'])
    
    # in batches backprop to update weights, forward prop using those weights to calculate loss for each epoch 
    metrics = model.fit(
        X_train, y_train,
        validation_data=(X_val, y_val),
        epochs=8,
        batch_size=batch_size
        )

    return metrics, model


