from typing import List, Iterator, Tuple, Any

import matplotlib.pyplot as plt
import numpy as np
from tensorflow.keras import layers
from tensorflow.keras.datasets import boston_housing
from tensorflow.python.keras import models


def load_data() -> Tuple[Tuple[Any, Any], Tuple[Any, Any]]:
    (train_data, train_targets), (test_data, test_targets) = boston_housing.load_data()

    mean = train_data.mean(axis=0)
    train_data -= mean

    std = train_data.std(axis=0)
    train_data /= std
    test_data -= mean
    test_data /= std

    return (train_data, train_targets), (test_data, test_targets)


def create_model(input_shape) -> models.Model:
    model = models.Sequential()

    model.add(layers.Dense(64, activation="relu", input_shape=(input_shape,)))
    model.add(layers.Dense(64, activation="relu"))
    model.add(layers.Dense(1))

    model.compile(optimizer="rmsprop", loss="mse",
                  metrics=["mae"])
    return model


def train_model(train_data: np.array, train_targets: np.array, validation_data: np.array,
                validation_targets: np.array, batch_size: int, epochs: int, k: int) -> Tuple[List[float], List[float], List[float], List[float]]:
    num_val_samples = len(train_data) // k
    all_mse = []
    all_val_mse = []
    all_mae = []
    all_val_mae = []

    for i in range(k):
        print("processing fold #", i)
        val_data = np.concatenate([train_data[i * num_val_samples: (i + 1) * num_val_samples], validation_data])
        val_targets = np.concatenate(
            [train_targets[i * num_val_samples: (i + 1) * num_val_samples], validation_targets])
        partial_train_data = np.concatenate([train_data[:i * num_val_samples], train_data[(i + 1) * num_val_samples:]],
                                            axis=0)
        partial_train_targets = np.concatenate(
            [train_targets[:i * num_val_samples], train_targets[(i + 1) * num_val_samples:]], axis=0)
        model = create_model(train_data.shape[1])
        history = model.fit(partial_train_data, partial_train_targets, validation_data=(val_data, val_targets),
                            epochs=epochs,
                            batch_size=batch_size, verbose=0).history
        mae = history["mae"]
        val_mae = history["val_mae"]
        mse = history["loss"]
        val_mse = history["val_loss"]
        draw_plot("Mae", range(1, epochs + 1), mae, val_mae)
        draw_plot("Mse", range(1, epochs + 1), mse, val_mse)
        all_mse.extend(mse)
        all_val_mse.extend(val_mse)
        all_mae.extend(mae)
        all_val_mae.extend(val_mae)

    return all_mse, all_val_mse, all_mae, all_val_mae


def draw_plot(data_type: str, epochs: Iterator[int], train_data_value: List[int],
              test_data_value: List[int]):
    plt.plot(epochs, train_data_value, "b", label=f"Training {data_type}")
    plt.plot(epochs, test_data_value, "r", label=f"Validation {data_type}")
    plt.title(f"Training and validation {data_type}")
    plt.xlabel("Epochs")
    plt.ylabel(f"{data_type}")
    plt.legend()
    plt.show()


def main():
    (train_data, train_targets), (test_data, test_targets) = load_data()
    k = 5
    epochs = 30

    all_mse, all_val_mse, all_mae, all_val_mae = train_model(train_data, train_targets, test_data, test_targets, 1, epochs, k)

    draw_plot("All Mse", range(1, epochs + 1), [np.mean(all_mse[i]) for i in range(epochs)],
              [np.mean(all_val_mse[i]) for i in range(epochs)])
    draw_plot("All Mae", range(1, epochs + 1), [np.mean(all_mae[i]) for i in range(epochs)],
              [np.mean(all_val_mae[i]) for i in range(epochs)])

    print(np.mean(all_val_mae))


if __name__ == "__main__":
    main()
