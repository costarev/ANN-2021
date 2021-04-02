import itertools
import math
from typing import Tuple, List, Callable

import numpy as np
from tensorflow.keras.layers import Dense
from tensorflow.keras.models import Sequential


def operation(a: bool, b: bool, c: bool) -> bool:
    return (a and not b) or ((c + b) % 2)


def naive_relu(matrix: np.array) -> np.array:
    assert len(matrix.shape) == 2
    matrix = matrix.copy()
    for i in range(matrix.shape[0]):
        for j in range(matrix.shape[1]):
            matrix[i, j] = max(matrix[i, j], 0)
    return matrix


def naive_add_matrix_and_vector(matrix: np.array, vector: np.array) -> np.array:
    assert len(matrix.shape) == 2
    assert len(vector.shape) == 1
    assert matrix.shape[1] == vector.shape[0]
    matrix = matrix.copy()
    for i in range(matrix.shape[0]):
        for j in range(matrix.shape[1]):
            matrix[i, j] += vector[j]
    return matrix


def naive_vector_dot(lhs_vector: np.array, rhs_vector: np.array) -> float:
    assert len(lhs_vector.shape) == 1
    assert len(rhs_vector.shape) == 1
    assert lhs_vector.shape[0] == rhs_vector.shape[0]
    sum_accumulator = 0.
    for i in range(lhs_vector.shape[0]):
        sum_accumulator += lhs_vector[i] * rhs_vector[i]
    return sum_accumulator


def naive_matrix_vector_dot(matrix: np.array, vector: np.array) -> np.array:
    assert len(matrix.shape) == 2
    assert len(vector.shape) == 1
    assert matrix.shape[1] == vector.shape[0]
    result_vector = np.zeros(matrix.shape[0])
    for i in range(matrix.shape[0]):
        for j in range(matrix.shape[1]):
            result_vector[i] += matrix[i, j] * vector[j]
    return result_vector


def naive_matrix_matrix_dot(lhs_matrix: np.array, rhs_matrix: np.array) -> np.array:
    assert len(lhs_matrix.shape) == 2
    assert len(rhs_matrix.shape) == 2
    assert lhs_matrix.shape[1] == rhs_matrix.shape[0]
    result_matrix = np.zeros((lhs_matrix.shape[0], rhs_matrix.shape[1]))
    for i in range(lhs_matrix.shape[0]):
        for j in range(rhs_matrix.shape[1]):
            for k in range(lhs_matrix.shape[1]):
                result_matrix[i, j] += lhs_matrix[i, k] * rhs_matrix[k, j]
    return result_matrix


def naive_sigmoid(vector: np.array) -> np.array:
    sigmoid_applied_vector = []
    for i in range(vector.shape[0]):
        sigmoid_applied_vector.append([1 / (1 + math.exp(-vector[i]))])
    return sigmoid_applied_vector


def make_naive_layer_iteration(data: np.array, layer: Dense, activation_fn: Callable) -> np.array:
    weights = layer.get_weights()[0]
    biases = layer.get_weights()[1]
    return activation_fn(naive_add_matrix_and_vector(naive_matrix_matrix_dot(data, weights), biases))


def naive_simulation(data: np.array, input_layers: List) -> np.array:
    layers = input_layers.copy()
    last_layer = layers.pop()
    for layer in layers:
        data = make_naive_layer_iteration(data, layer, naive_relu)
    return make_naive_layer_iteration(data, last_layer, naive_sigmoid)


def make_np_layer_iteration(data: np.array, layer: Dense, activation_fn: Callable) -> np.array:
    weights = layer.get_weights()[0]
    biases = layer.get_weights()[1]
    return activation_fn(np.dot(data, weights) + biases)


def np_simulation(data: np.array, input_layers: List) -> np.array:
    layers = input_layers.copy()
    last_layer = layers.pop()
    for layer in layers:
        data = make_np_layer_iteration(data, layer, lambda x: np.maximum(x, 0))
    return make_np_layer_iteration(data, last_layer, lambda x: 1 / (1 + np.exp(-x)))


def build_model() -> Sequential:
    sequential_model = Sequential()
    sequential_model.add(Dense(16, activation='relu', input_shape=(3,)))
    sequential_model.add(Dense(8, activation='relu'))
    sequential_model.add(Dense(16, activation='relu'))
    sequential_model.add(Dense(8, activation='relu'))
    sequential_model.add(Dense(1, activation='sigmoid'))
    return sequential_model


def simulation_logging(data: np.array, model: Sequential):
    print('Naive', *np.round(naive_simulation(data, model.layers), 3), sep='\t')
    print('Numpy', *np.round(np_simulation(data, model.layers), 3), sep='\t')
    print('Keras', *np.round(model.predict(data), 3), sep='\t')
    print('-----------')
    print('Naive rounded', *np.round(naive_simulation(data, model.layers), 0), sep='\t')
    print('Numpy rounded', *np.round(np_simulation(data, model.layers), 0), sep='\t')
    print('Keras rounded', *np.round(model.predict(data), 0), sep='\t')
    print("-----------------------------------")


def generate_sample(feature_count=3) -> Tuple[np.array, np.array]:
    feature_rows = []
    correct_values = []
    for binary_sequence in itertools.product([0, 1], repeat=feature_count):
        feature_rows.append(binary_sequence)
        correct_values.append(operation(*binary_sequence))
    return np.array(feature_rows), np.array(correct_values)


if __name__ == '__main__':
    X, y = generate_sample(3)
    model = build_model()
    print('Before fitting')
    simulation_logging(X, model)
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    model.fit(X, y, epochs=100, batch_size=1, verbose=False)
    print('After fitting:\n')
    simulation_logging(X, model)
    print('Correct: ', *y, sep='\t')
