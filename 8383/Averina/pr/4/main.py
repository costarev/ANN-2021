import numpy as np
from tensorflow.keras import layers
from tensorflow.keras import models

# Variant 1
# (a and b) or (a and c)


def logic_func(x):
    return (x[0] and x[1]) or (x[0] and x[2])


def naive_relu(x):
    assert len(x.shape) == 2
    x = x.copy()
    for i in range(x.shape[0]):
        for j in range(x.shape[1]):
            x[i, j] = max(x[i, j], 0)
    return x


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
            row_x = x[i, :]
            column_y = y[:, j]
            z[i, j] = naive_vector_dot(row_x, column_y)
    return z


def naive_simulation(layers, data):
    data = naive_relu(
        naive_matrix_matrix_dot(data, layers[0].get_weights()[0]) + layers[0].get_weights()[1])
    data = naive_relu(
        naive_matrix_matrix_dot(data, layers[1].get_weights()[0]) + layers[1].get_weights()[1])
    data = sigmoid(
        naive_matrix_matrix_dot(data, layers[2].get_weights()[0]) + layers[2].get_weights()[1])
    return data


def numpy_simulation(layers, data):
    data = np.maximum(
        np.dot(data, layers[0].get_weights()[0]) + layers[0].get_weights()[1], 0)
    data = np.maximum(
        np.dot(data, layers[1].get_weights()[0]) + layers[1].get_weights()[1], 0)
    data = sigmoid(
        np.dot(data, layers[2].get_weights()[0]) + layers[2].get_weights()[1])
    return data


train_data = np.genfromtxt('input.csv', delimiter=',')
train_label = np.asarray([logic_func(x) for x in train_data])
print(train_label)

model = models.Sequential()
model.add(layers.Dense(8, activation='relu', input_shape=(3,)))
model.add(layers.Dense(16, activation='relu'))
model.add(layers.Dense(1, activation='sigmoid'))
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

print("Before training:")
print("Model.predict:")
print(model.predict(train_data))

print("Naive:\n", naive_simulation(model.layers, train_data))
print("Numpy:\n", numpy_simulation(model.layers, train_data))

H = model.fit(train_data, train_label, epochs=700, verbose=False)

print("After training:")
print("model.predict:")
print(model.predict(train_data))

print("Naive:\n", naive_simulation(model.layers, train_data))
print("Numpy:\n", numpy_simulation(model.layers, train_data))
