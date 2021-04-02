from tensorflow.keras.layers import Dense
from tensorflow.keras.models import Sequential
import numpy as np

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

def naive_simulation(layers, input):
    output1 = naive_relu(naive_matrix_matrix_dot(input, layers[0].get_weights()[0]) + layers[0].get_weights()[1])
    output2 = naive_relu(naive_matrix_matrix_dot(output1, layers[1].get_weights()[0]) + layers[1].get_weights()[1])
    res = sigmoid(naive_matrix_matrix_dot(output2, layers[2].get_weights()[0]) + layers[2].get_weights()[1])
    return np.reshape(res, ((input.shape[0]), 1))

def np_simulation(layers, input):
    output1 = np.maximum(np.dot(input, layers[0].get_weights()[0]) + layers[0].get_weights()[1], 0)
    output2 = np.maximum(np.dot(output1, layers[1].get_weights()[0]) + layers[1].get_weights()[1], 0)
    res = sigmoid(np.dot(output2, layers[2].get_weights()[0]) + layers[2].get_weights()[1])
    return np.reshape(res, ((input.shape[0]), 1))


model = Sequential()
model.add(Dense(8, activation='relu', input_shape=(3,)))
model.add(Dense(32, activation='relu'))
model.add(Dense(1, activation='sigmoid'))
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

train_data = np.array(
                    [[0, 0, 0],
                    [0, 0, 1],
                    [0, 1, 0],
                    [0, 1, 1],
                    [1, 0, 0],
                    [1, 0, 1],
                    [1, 1, 0],
                    [1, 1, 1]])
train_labels = np.array([[0],[1],[1],[0],[1],[1],[1],[0]])

print("Numpy untrained simulation:\n", np_simulation(model.layers, train_data))
print("Naive untrained simulation:\n", naive_simulation(model.layers, train_data))
print("Untrained model predict:\n", model.predict(train_data))

history = model.fit(train_data, train_labels, epochs=400, verbose=False)

print("Numpy trained simulation:\n", np_simulation(model.layers, train_data))
print("Naive untrained simulation:\n", naive_simulation(model.layers, train_data))
print("Trained model predict:\n", model.predict(train_data))