import math

import numpy as np
from tensorflow.keras.layers import Dense
from tensorflow.keras.models import Sequential


# (a or b) xor not(b and c)


def operation(x):
    return (x[0] or x[1]) != (not (x[1] and x[2]))


def naive_relu(x):
    assert len(x.shape) == 2  # проверка размерности 2
    x = x.copy()  # копирования от защиты изменения исходного тензора
    for i in range(x.shape[0]):
        for j in range(x.shape[1]):
            x[i, j] = max(x[i, j], 0)
    return x


def naive_add_matrix_and_vector(x, y):
    assert len(x.shape) == 2.
    assert len(y.shape) == 1.
    assert x.shape[1] == y.shape[0]
    x = x.copy()
    for i in range(x.shape[0]):
        for j in range(x.shape[1]):
            x[i, j] += y[j]
    return x


def naive_vector_dot(x, y):
    assert len(x.shape) == 1
    assert len(y.shape) == 1
    assert x.shape[0] == y.shape[0]
    z = 0.
    for i in range(x.shape[0]):
        z += x[i] * y[i]
    return z


def naive_matrix_vector_dot(x, y):
    assert len(x.shape) == 2
    assert len(y.shape) == 1
    assert x.shape[1] == y.shape[0]
    z = np.zeros(x.shape[0])
    for i in range(x.shape[0]):
        for j in range(x.shape[1]):
            z[i] += x[i, j] * y[j]
    return z


def naive_matrix_matrix_dot(x, y):
    assert len(x.shape) == 2 # размерность первого тензора 2
    assert len(y.shape) == 2 # размерность второго тензора 2
    assert x.shape[1] == y.shape[0] # проверка возможности перемножения тензоров
    z = [[0 for _ in range(y.shape[1])] for _ in range(x.shape[0])]
    for i in range(x.shape[0]):
        for j in range(y.shape[1]):
            cur_row = x[i, :]
            cur_col = y[:, j]
            z[i][j] = naive_vector_dot(cur_row, cur_col)
    return np.array(z)


def naive_sigmoid(x):
    y = []
    for x_i in x:
        y.append([1 / (1 + math.exp(-x_i))])
    return y


def naive_sim(layers, data):
    for layer in layers[:-1]:
        weights = layer.get_weights()[0]
        biases = layer.get_weights()[1]
        data = naive_relu(naive_add_matrix_and_vector(naive_matrix_matrix_dot(data, weights), biases))
    weights = layers[-1].get_weights()[0]
    biases = layers[-1].get_weights()[1]
    data = naive_sigmoid(naive_add_matrix_and_vector(naive_matrix_matrix_dot(data, weights), biases))
    return data


def np_sim(layers, data):
    for layer in layers[:-1]:
        weights = layer.get_weights()[0]
        biases = layer.get_weights()[1]
        data = np.maximum(np.dot(data, weights) + biases, 0)
    weights = layers[-1].get_weights()[0]
    biases = layers[-1].get_weights()[1]
    data = 1 / (1 + np.exp(-(np.dot(data, weights) + biases)))
    return data


def build_model():
    model = Sequential()
    model.add(Dense(32, activation='relu', input_shape=(3,)))
    model.add(Dense(16, activation='relu'))
    model.add(Dense(8, activation='relu'))
    model.add(Dense(1, activation='sigmoid'))
    return model


def print_res(m, data):
    print('Keras', *np.round(m.predict(data), 7), sep='\t')
    print('Naive', *np.round(naive_sim(m.layers, data), 7), sep='\t')
    print('Numpy', *np.round(np_sim(m.layers, data), 7), sep='\t')
    print('After round')
    print('Keras', *np.round(m.predict(data), 0), sep='\t')
    print('Naive', *np.round(naive_sim(m.layers, data), 0), sep='\t')
    print('Numpy', *np.round(np_sim(m.layers, data), 0), sep='\t')


if __name__ == '__main__':
    X = np.array([
        [0, 0, 0],  # 1
        [0, 0, 1],  # 1
        [0, 1, 0],  # 0
        [0, 1, 1],  # 1
        [1, 0, 0],  # 0
        [1, 0, 1],  # 0
        [1, 1, 0],  # 0
        [1, 1, 1],  # 1
    ])
    y = np.array([operation(x_) for x_ in X])
    model = build_model()
    print('Before fitting')
    print_res(model, X)
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    model.fit(X, y, epochs=64, batch_size=4, verbose=False)
    print('After fitting')
    print_res(model, X)
    print('Real', *[operation(x_) for x_ in X], sep='\t')
