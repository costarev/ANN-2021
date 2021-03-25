from tensorflow.keras.layers import Dense
from tensorflow.keras.models import Sequential
import numpy as np


def logical_operation(a, b, c):
    return (a and b and c) ^ (a or not b)


def relu(x):
    return np.maximum(x, 0)


def sigmoid(x):
    return 1 / (1 + np.exp(-x))


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
            z[i, j] += naive_vector_dot(x[i, :], y[:, j])
    return z


def elements_operations(data, layers):
    for i in range(len(layers) - 1):
        data = relu(naive_matrix_matrix_dot(data, layers[i].get_weights()[0]) + layers[i].get_weights()[1])

    data = sigmoid(naive_matrix_matrix_dot(data, layers[len(layers)-1].get_weights()[0]) + layers[len(layers)-1].get_weights()[1])
    return data


def numpy_operations(data, layers):
    for i in range(len(layers) - 1):
        data = np.maximum(np.dot(data, layers[i].get_weights()[0]) + layers[i].get_weights()[1], 0)

    data = sigmoid(np.dot(data, layers[len(layers)-1].get_weights()[0]) + layers[len(layers)-1].get_weights()[1])
    return data


def create_model():
    model = Sequential()
    model.add(Dense(8, activation='relu', input_dim=3))
    model.add(Dense(32, activation='relu'))
    model.add(Dense(1, activation='sigmoid'))
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    return model


data = np.reshape(np.fromfile('data.txt', dtype='int', sep=' '), (-1, 3))
logical_result = np.array([int(logical_operation(row[0], row[1], row[2])) for row in data])
print("Real result:")
print(logical_result)

model = create_model()

print("Elementwise function:")
print(elements_operations(data, model.layers))
print("Numpy function:")
print(numpy_operations(data, model.layers))

print('Training...')
model.fit(data, logical_result, epochs=500, verbose=0)

print("Elementwise function:")
print(elements_operations(data, model.layers))
print("Numpy function:")
print(numpy_operations(data, model.layers))