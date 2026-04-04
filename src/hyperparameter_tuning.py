import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
import os

from src.model_utils import train_and_validate_model, split_data

from src.supabase import extract_imgs_from_db

main_bucket = os.environ.get("MAIN_BUCKET")

def minimise_validation_loss(metrics):

    validation_losses = metrics.history['val_loss']
    training_losses = metrics.history['loss']

    num_epochs = len(validation_losses)
    print('num epochs', num_epochs)
    x = np.arange(1, num_epochs + 1, 1)

    plt.figure(figsize=(10, 6))

    plt.plot(x, validation_losses, label="validation_loss")
    plt.plot(x, training_losses, label="training_loss")

    plt.xticks(x)

    plt.xlabel("Number of epochs")
    plt.ylabel("Loss")
    plt.title("Validation loss vs number of epochs")
    plt.legend()

    plt.show()

def batch_size_tuning():

    images = extract_imgs_from_db(main_bucket)

    X_train, y_train, X_val, y_val, X_test, y_test = split_data(images)

    x = np.arange(1, 9, 1)

    plt.figure(figsize=(10, 6))

    for batch_size in (8, 16, 32, 64): 
        metrics = train_and_validate_model(X_train, y_train, X_val, y_val, X_test, y_test, batch_size)

        validation_losses = metrics.history['val_loss']

        plt.plot(x, validation_losses, label=f'batch size {batch_size}')

    plt.xticks(x)

    plt.xlabel("Number of epochs for various batch sizes")
    plt.ylabel("Loss")
    plt.title("Validation loss ")
    plt.legend()

    plt.show()

def learning_rate_tuning():
    
    images = extract_imgs_from_db(main_bucket)

    X_train, y_train, X_val, y_val, X_test, y_test = split_data(images)

    x = np.arange(1, 9, 1)

    plt.figure(figsize=(10, 6))

    for learning_rate in (0.001, 0.005, 0.0001, 0.01): 
        adam = tf.keras.optimizers.Adam(learning_rate=learning_rate)
        metrics = train_and_validate_model(X_train, y_train, X_val, y_val, X_test, y_test, 32, adam)

        validation_losses = metrics.history['val_loss']

        plt.plot(x, validation_losses, label=f'learning rate {learning_rate}')

    plt.xticks(x)

    plt.xlabel("Number of epochs for various learning rates")
    plt.ylabel("Loss")
    plt.title("Validation loss")
    plt.legend()

    plt.show()

if __name__ == "__main__":
    learning_rate_tuning()

