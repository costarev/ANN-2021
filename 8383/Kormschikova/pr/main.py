# (a or b) xor not(b and c)
import numpy as np
import pandas
from tensorflow.keras.layers import Dense
from tensorflow.keras.models import Sequential


def sigmoid(x):
    return 1 / (1 + np.exp(-x))


def naive_relu(x):
    assert len(x.shape) == 2  # проверка размерности 2
    x = x.copy()  # копирования от защиты изменения исходного тензора
    for i in range(x.shape[0]):
        for j in range(x.shape[1]):
            x[i, j] = max(x[i, j], 0)
    return x


def naive_vector_dot(x, y):
    assert len(x.shape) == 1
    assert len(y.shape) == 1
    assert x.shape[0] == y.shape[0]
    z = 0.
    for i in range(x.shape[0]):
        z += x[i] * y[i]
    return z


def naive_matrix_matrix_dot(x, y):
    assert len(x.shape) == 2
    assert len(y.shape) == 2
    assert x.shape[1] == y.shape[0]

    z = np.zeros((x.shape[0], y.shape[1]))
    for i in range(x.shape[0]):
        for j in range(y.shape[1]):
            z[i, j] = naive_vector_dot(x[i, :], y[:, j])
    return z


def naive_simulation(layers, input):
    lenL = len(layers) - 1;
    output = input;
    for i in range(lenL):
        output = naive_relu(naive_matrix_matrix_dot(output, layers[i].get_weights()[0]) + layers[i].get_weights()[1])
    output = sigmoid(naive_matrix_matrix_dot(output, layers[lenL].get_weights()[0]) + layers[lenL].get_weights()[1])
    return np.reshape(output, ((input.shape[0]), 1))

def numpy_simulation(layers, input):
    lenL = len(layers) - 1;
    output = input;
    for i in range(lenL):
        output = np.maximum(np.dot(output, layers[i].get_weights()[0]) + layers[i].get_weights()[1], 0)
    output = sigmoid(np.dot(output, layers[lenL].get_weights()[0]) + layers[lenL].get_weights()[1])
    return np.reshape(output, ((input.shape[0]), 1))


dataframe = pandas.read_csv("input.csv", header = None, sep=";")
dataset = dataframe.values
X = dataset[:, 0:3].astype(int)  # входные данные
Y = dataset[:, 3].astype(int)  # выходные данные


model = Sequential()
model.add(Dense(8, activation='relu', input_shape=(3,)))
model.add(Dense(8, activation='relu'))
model.add(Dense(1, activation='sigmoid'))
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

print("------------Untrained------------\n")
print("Model:\n", model.predict(X))
print("Numpy:\n", numpy_simulation(model.layers, X))
print("Naive:\n", naive_simulation(model.layers, X))

model.fit(X, Y, epochs=200, verbose=0)
print("------------Trained------------\n ")
print("Model:\n", model.predict(X))
print("Numpy:\n", numpy_simulation(model.layers, X))
print("Naive:\n", naive_simulation(model.layers, X))
