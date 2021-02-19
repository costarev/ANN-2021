import pandas as pd
from tensorflow.keras.layers import Dense
from tensorflow.keras.models import Sequential
from tensorflow.keras.utils import to_categorical
from sklearn.preprocessing import LabelEncoder
import seaborn as sns
import matplotlib.pyplot as plt
from collections import namedtuple
from time import time


def model_configurator(X: pd.array, y: pd.array, hidden_layers: list, epochs: int, batch_size: int,
                       validation_split: float):
    model = Sequential()
    model.add(Dense(4, activation='relu'))
    for layer in hidden_layers:
        model.add(Dense(layer.neuron_count, activation=layer.activation))
    model.add(Dense(3, activation='softmax'))
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    return model.fit(X, y, epochs=epochs, batch_size=batch_size, validation_split=validation_split, verbose=False)


def plot(data: pd.DataFrame, label: str, title: str):
    axis = sns.lineplot(data=data, dashes=False)
    axis.set(ylabel=label, xlabel='epochs', title=title)
    axis.grid(True, linestyle="--")
    plt.show()
    # plt.savefig(f"img/{title[-1]}/{title.replace(' ', '_')}_{label}_{int(time())}")
    # plt.clf()


if __name__ == '__main__':
    Layer = namedtuple('Layer', 'neuron_count activation')
    dataframe = pd.read_csv("iris.csv", header=None)
    dataset = dataframe.values
    X = dataset[:, 0:4].astype(float)
    Y = dataset[:, 4]
    encoder = LabelEncoder()
    encoder.fit(Y)
    encoded_Y = encoder.transform(Y)
    dummy_y = to_categorical(encoded_Y)

    params = [
        [X, dummy_y, [], 200, 10, 0.1],
        [X, dummy_y, [], 200, 10, 0.2],
        [X, dummy_y, [], 200, 10, 0.6],

        [X, dummy_y, [], 200, 3, 0.1],
        [X, dummy_y, [], 200, 30, 0.1],
        [X, dummy_y, [Layer(8, "relu")], 200, 3, 0.1],

        [X, dummy_y, [Layer(8, "relu"), Layer(8, "relu")], 200, 3, 0.1],
        [X, dummy_y, [Layer(16, "relu")], 200, 3, 0.1],
        [X, dummy_y, [Layer(16, "relu"), Layer(16, "relu")], 200, 3, 0.1],
    ]
    for i in range(len(params)):
        fitted = model_configurator(*params[i])
        df_hist = pd.DataFrame(fitted.history)
        plot(df_hist[['loss', 'val_loss']], "Loss", "ANN #" + str(i+1))
        plot(df_hist[['accuracy', 'val_accuracy']], "Accuracy", "ANN #" + str(i+1))
