import pandas

from tensorflow.keras.layers import Dense
from tensorflow.keras.models import Sequential
from tensorflow.keras.utils import to_categorical
from sklearn.preprocessing import LabelEncoder
import matplotlib.pyplot as plot


def plot_single_history(history, color="blue"):
    keys = ["loss", "accuracy", "val_loss", "val_accuracy"]
    titles = ["Loss", "Accuracy", "Val loss", "Val accuracy"]
    xlabels = ["epoch", "epoch", "epoch", "epoch"]
    ylabels = ["loss", "accuracy", "loss", "accuracy"]
    ylims = [3, 1.1, 3, 1.1]

    for i in range(len(keys)):
        plot.subplot(2, 2, i + 1)
        plot.title(titles[i])
        plot.xlabel(xlabels[i])
        plot.ylabel(ylabels[i])
        plot.gca().set_ylim([0, ylims[i]])
        plot.grid()
        values = history[keys[i]]
        plot.plot(range(len(values)), values, color=color)


def execute_model(layers=[], epochs=75, batch_size=10, validation_split=0.1):
    model = Sequential()
    model.add(Dense(4, activation='relu'))

    for layer in layers:
        model.add(layer)

    model.add(Dense(3, activation='softmax'))

    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

    return model.fit(
        X, dummy_y,
        epochs=epochs,
        batch_size=batch_size,
        validation_split=validation_split
    ).history


def run_layers_tests(tests):
    for i in range(len(tests)):
        history = execute_model(layers=tests[i], epochs=100)
        plot_single_history(history, layersColors[i])


def run_params_tests(tests, layers_test):
    for i in range(len(tests)):
        test = tests[i]
        history = execute_model(layers=layers_test, epochs=100, batch_size=test["batch"], validation_split=test["validation_split"])
        plot_single_history(history, layersColors[i])


dataframe = pandas.read_csv("iris.csv", header=None)

dataset = dataframe.values

X = dataset[:, 0:4].astype(float)
Y = dataset[:, 4]

encoder = LabelEncoder()
encoder.fit(Y)
encoded_Y = encoder.transform(Y)
dummy_y = to_categorical(encoded_Y)

layersColors = [
    "red",
    "blue",
    "purple",
    "green",
]

# тесты количества нейронов relu
layersTests1Title = "relu neurons"
layersTests1 = [
    [
        Dense(64, activation='relu'),
    ],
    [
        Dense(192, activation='relu'),
    ],
]

# тесты количества нейронов softmax
layersTests2 = [
    [
        Dense(64, activation='sigmoid'),
    ],
    [
        Dense(192, activation='sigmoid'),
    ],
]

layersTests3 = [
    [
        Dense(64, activation='relu'),
        Dense(64, activation='relu'),
    ],
    [
        Dense(192, activation='relu'),
        Dense(192, activation='sigmoid'),
    ],
]

layersTests4 = [
    [
        Dense(64, activation='relu'),
        Dense(64, activation='relu'),
    ],
    [
        Dense(64, activation='relu'),
        Dense(64, activation='relu'),
        Dense(64, activation='relu'),
    ],
]

paramsTests = [
    {
        "batch": 10,
        "validation_split": 0.1,
    },
    {
        "batch": 30,
        "validation_split": 0.1,
    },
]

#run_layers_tests([[]])
#run_layers_tests(layersTests2)
run_params_tests(paramsTests, layersTests3[0])

plot.show()
