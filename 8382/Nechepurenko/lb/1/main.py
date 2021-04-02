import pandas as pd
import numpy as np
from tensorflow.keras.utils import to_categorical
from sklearn.preprocessing import LabelEncoder
import matplotlib.pyplot as plt

# %matplotlib inline
from lb1.models import ModelBuilder, get_models_to_compare


def read_data():
    dataframe = pd.read_csv("sample_data/iris.csv", header=None)
    dataset = dataframe.values
    np.random.seed(42)
    np.random.shuffle(dataset)
    x = dataset[:, 0:4].astype(float)
    y = dataset[:, 4]
    return x, y


def encode_y(raw_y):
    encoder = LabelEncoder()
    encoder.fit(raw_y)
    encoded_y = encoder.transform(raw_y)
    y_mapping = dict(zip(encoder.classes_, encoder.transform(encoder.classes_)))
    return to_categorical(encoded_y), y_mapping


def plot(epochs, train, validation, metrics):
    plt.plot(epochs, train, 'b', label=f'Training {metrics}')
    plt.plot(epochs, validation, 'r', label=f'Validation {metrics}')
    plt.title(f'Training and validation {metrics}')
    plt.xlabel('Epochs')
    plt.ylabel(metrics.capitalize())
    plt.grid(True)
    plt.legend()


def summary(h):
    print(f"Model #{h.idx + 1} has best val_accuracy {max(h.history.get('val_accuracy'))}")
    # maybe next time


def plot_history(history):
    loss = history['loss']
    val_loss = history['val_loss']
    acc = history['accuracy']
    val_acc = history['val_accuracy']
    epochs = range(1, len(loss) + 1)

    plt.figure()
    plt.subplot(211)
    plot(epochs, loss, val_loss, "loss")
    plt.subplot(212)
    plot(epochs, acc, val_acc, "accuracy")
    plt.show()


X, Y = read_data()
Y, mapping = encode_y(Y)

history_list = []
for model in get_models_to_compare():
    model_cfg, model_layers = model
    model_builder = ModelBuilder()
    model_builder.set_params(**model_cfg)
    model = model_builder.set_layers(model_layers).build()
    history_list.append(model.fit(X, Y))

for idx, history_item in enumerate(history_list):
    history_item.idx = idx
    summary(history_item)
    plot_history(history_item.history)
