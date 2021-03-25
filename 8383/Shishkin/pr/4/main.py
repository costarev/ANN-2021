import numpy as np
from tensorflow.keras import layers
from tensorflow.keras import models


# Var. 8: (a and c and b) xor (a or not b)


def my_logic_operation(x):
    return (x[0] and x[2] and x[1]) != (x[0] or not x[1])


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


def naive_simulation(model_layers, train_data_f, numOfLayers):
    for i in range(numOfLayers - 1):
        train_data_f = naive_relu(naive_matrix_matrix_dot(train_data_f, model_layers[i].get_weights()[0])
                                  + model_layers[i].get_weights()[1])
    train_data_f = sigmoid(naive_matrix_matrix_dot(train_data_f, model_layers[numOfLayers - 1].get_weights()[0])
                           + model_layers[numOfLayers - 1].get_weights()[1])

    return train_data_f


def numpy_simulation(model_layers, train_data_f, numOfLayers):
    for i in range(numOfLayers - 1):
        train_data_f = np.maximum(np.dot(train_data_f, model_layers[i].get_weights()[0])
                                  + model_layers[i].get_weights()[1], 0)
    train_data_f = sigmoid(np.dot(train_data_f, model_layers[numOfLayers - 1].get_weights()[0])
                           + model_layers[numOfLayers - 1].get_weights()[1])

    return train_data_f


train_data = np.asarray([[0, 0, 0], [0, 0, 1], [0, 1, 0], [0, 1, 1], [1, 0, 0], [1, 0, 1], [1, 1, 0], [1, 1, 1]])
train_label = np.asarray([my_logic_operation(x) for x in train_data])
print(train_label)

numOfLayers = 3
model = models.Sequential()
model.add(layers.Dense(16, activation='relu', input_shape=(3,)))
model.add(layers.Dense(8, activation='relu'))
model.add(layers.Dense(1, activation='sigmoid'))

print("Before training:")
print("model.predict:")
print(model.predict(train_data))
naive_res_before = naive_simulation(model.layers, train_data, numOfLayers)
numpy_res_before = numpy_simulation(model.layers, train_data, numOfLayers)

print("Naive:")
print(naive_res_before)
print("Naive coincidence before:")
for i in range(train_label.shape[0]):
    print(np.round(naive_res_before[i]) == train_label[i])

print("Numpy:")
print(numpy_res_before)
print("Numpy coincidence before:")
for i in range(train_label.shape[0]):
    print(np.round(numpy_res_before[i]) == train_label[i])

model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
H = model.fit(train_data, train_label, epochs=800, verbose=False)

print("After training:")
print("model.predict:")
print(model.predict(train_data))
naive_res_after = naive_simulation(model.layers, train_data, numOfLayers)
numpy_res_after = numpy_simulation(model.layers, train_data, numOfLayers)

print("Naive:")
print(naive_res_after)
print("Naive coincidence after:")
for i in range(train_label.shape[0]):
    print(np.round(naive_res_after[i]) == train_label[i])

print("Numpy:")
print(numpy_res_after)
print("Numpy coincidence after:")
for i in range(train_label.shape[0]):
    print(np.round(numpy_res_after[i]) == train_label[i])
