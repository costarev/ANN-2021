import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from tensorflow.keras.layers import Dense
from tensorflow.keras.models import Sequential
from collections import namedtuple

Layer = namedtuple('Layer', 'neuron_count activation')


def model_configurator(X: pd.array, y: pd.array, layers: list, epochs: int, batch_size: int, validation_split: float):
    model = Sequential()
    model.add(Dense(layers[0].neuron_count, input_shape=(layers[0].neuron_count,), activation=layers[0].activation))
    for layer in layers[1:]:
        model.add(Dense(layer.neuron_count, activation=layer.activation))
    model.add(Dense(1, activation='sigmoid'))
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    return model.fit(X[:, :layers[0].neuron_count], y, epochs=epochs, batch_size=batch_size,
                     validation_split=validation_split, verbose=False)


def plot(data: pd.DataFrame, label: str, title: str):
    axis = sns.lineplot(data=data, dashes=False)
    axis.set(ylabel=label, xlabel='epochs', title=title.split('.')[0])
    axis.grid(True, linestyle="--")
    # plt.show()
    plt.savefig(f"img/{title.replace(' ', '_').replace('#', '').replace('.', '-')}_{label}")
    plt.clf()


if __name__ == '__main__':
    np.random.seed(42)
    data = pd.read_csv("sonar.csv", header=None)
    np.random.shuffle(data.values)
    X = data.loc[:, 0:59].values
    y = pd.get_dummies(data.loc[:, 60]).drop('R', axis=1).values
    params = [
        [X, y, [Layer(60, "relu")], 100, 10, 0.1],
        [X, y, [Layer(30, "relu")], 100, 10, 0.1],
        [X, y, [Layer(30, "relu"), Layer(15, "relu")], 100, 10, 0.1],
        [X, y, [Layer(60, "relu"), Layer(30, "relu")], 100, 10, 0.1],
        [X, y, [Layer(30, "relu"), Layer(15, "relu"), Layer(15, "relu")], 100, 10, 0.1],
    ]
    for i in range(len(params)):
        for j in range(3):
            fitted = model_configurator(*params[i])
            df_history = pd.DataFrame(fitted.history)
            plot(df_history[['loss', 'val_loss']], "Loss", f"ANN #{i + 1}.{j}")
            plot(df_history[['accuracy', 'val_accuracy']], "Accuracy", f"ANN #{i + 1}.{j}")
