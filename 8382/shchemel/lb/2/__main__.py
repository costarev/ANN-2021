from typing import List, Iterator, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas
from sklearn.preprocessing import LabelEncoder
from tensorflow.keras import layers
from tensorflow.python.keras import models
from tensorflow.python.keras.callbacks import History


def load_data(filename: str) -> Tuple[List[int], List[int]]:
    dataframe = pandas.read_csv(filename, header=None)
    dataset = dataframe.values
    data = dataset[:, :60].astype(float)
    string_labels = dataset[:, 60]
    encoder = LabelEncoder()
    encoder.fit(string_labels)
    encoded_labels = encoder.transform(string_labels)
    return data, encoded_labels


def create_model() -> models.Model:
    model = models.Sequential()

    model.add(layers.Dense(60, activation="relu"))
    model.add(layers.Dense(15, activation="relu"))
    model.add(layers.Dense(1, activation="sigmoid"))

    model.compile(optimizer="adam", loss="binary_crossentropy",
                  metrics=["accuracy"])
    return model


def train_model(model: models.Model, data: np.array, labels: np.array, batch_size: int, epochs: int,
                validation_split: float) -> History:
    return model.fit(data, labels, batch_size=batch_size, epochs=epochs, validation_split=validation_split)


def draw_plot_for(data_type: str, epochs: Iterator[int], train_data_value: List[int], test_data_value: List[int]):
    plt.plot(epochs, train_data_value, "bo", label=f"Training {data_type}")
    plt.plot(epochs, test_data_value, "b", label=f"Validation {data_type}")
    plt.title(f"Training and validation {data_type}")
    plt.xlabel("Epochs")
    plt.ylabel(f"{data_type}")
    plt.legend()
    plt.show()


def main():
    data, labels = load_data("sonar.scv")

    model = create_model()
    history = train_model(model, data, labels, 10, 100, 0.1).history

    loss = history["loss"]
    val_loss = history["val_loss"]
    acc = history["accuracy"]
    val_acc = history["val_accuracy"]
    epochs = range(1, len(loss) + 1)

    draw_plot_for("Loss", epochs, loss, val_loss)

    plt.clf()
    draw_plot_for("Accuracy", epochs, acc, val_acc)

    results = model.evaluate(data, labels)
    print(results)


if __name__ == "__main__":
    main()
