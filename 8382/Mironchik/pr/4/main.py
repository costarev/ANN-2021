import numpy as np
from tensorflow.keras.layers import Dense
from tensorflow.keras.models import Sequential


def sigmoid(x):
    return 1 / (1 + np.exp(-x))


def naive_relu(x):
    assert len(x.shape) == 2
    x = x.copy()
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


def naive_run(layers, input):
    res = naive_relu(naive_matrix_matrix_dot(input, layers[0].get_weights()[0]) + layers[0].get_weights()[1])
    res = naive_relu(naive_matrix_matrix_dot(res, layers[1].get_weights()[0]) + layers[1].get_weights()[1])
    res = sigmoid(naive_matrix_matrix_dot(res, layers[2].get_weights()[0]) + layers[2].get_weights()[1])
    return res


def np_run(layers, input):
    res = np.maximum(np.dot(input, layers[0].get_weights()[0]) + layers[0].get_weights()[1], 0)
    res = np.maximum(np.dot(res, layers[1].get_weights()[0]) + layers[1].get_weights()[1], 0)
    res = sigmoid(np.dot(res, layers[2].get_weights()[0]) + layers[2].get_weights()[1])
    return res


model = Sequential()
model.add(Dense(16, activation='relu', input_shape=(3,)))
model.add(Dense(16, activation='relu'))
model.add(Dense(1, activation='sigmoid'))
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

data = np.genfromtxt('input_data.csv', delimiter=';')
train_data = data[:, :3]
train_labels = data[:, 3]

print("Untrained --------")
print("Numpy:\n", np_run(model.layers, train_data))
print("Naive:\n", naive_run(model.layers, train_data))
print("Model predict:\n", model.predict(train_data))

history = model.fit(train_data, train_labels, epochs=600, verbose=False)

print("Trained ---------")
print("Numpy:\n", np_run(model.layers, train_data))
print("Naive:\n", naive_run(model.layers, train_data))
print("Model predict:\n", model.predict(train_data))
