import numpy as np
from tensorflow.keras import layers
from tensorflow.keras import models


# (a or b) xor not(b and c)
def boolean_function(a, b, c):
    return (a or b) != (not (b and c))


def relu(x):
    return np.maximum(x, 0)


def sigmoid(x):
    return 1 / (1 + np.exp(-x))


def naive_add_matrix_and_vector(x, y):
    assert len(x.shape) == 2.
    assert len(y.shape) == 1.
    assert x.shape[1] == y.shape[0]
    x = x.copy()
    for i in range(x.shape[0]):
        for j in range(x.shape[1]):
            x[i, j] += y[j]
    return x


def naive_matrix_vector_dot(x, y):
    assert len(x.shape) == 2
    assert len(y.shape) == 1
    assert x.shape[1] == y.shape[0]
    z = np.zeros(x.shape[0])
    for i in range(x.shape[0]):
        for j in range(x.shape[1]):
            z[i] += x[i, j] * y[j]
    return z


def np_predict(data, weights):
    data = data.copy()

    functions = [relu, relu, sigmoid]
    for i in range(len(functions)):
        data = np.dot(data, np.asarray(weights[i][0]))
        data += np.asarray(weights[i][1])
        data = functions[i](data)
    return data


def naive_predict(data, weights):
    data = data.copy()

    functions = [relu, relu, sigmoid]
    for i in range(len(functions)):
        data = np.asarray([naive_matrix_vector_dot(np.asarray(weights[i][0]).transpose(), line) for line in data])
        data = naive_add_matrix_and_vector(data, np.asarray(weights[i][1]))
        data = np.asarray([functions[i](line) for line in data])
    return data


data = []
label = []
for a in [0, 1]:
    for b in [0, 1]:
        for c in [0, 1]:
            data.append([a, b, c])
            label.append(boolean_function(a, b, c))
data = np.asarray(data)
label = np.asarray(label)


model = models.Sequential()
model.add(layers.Dense(12, activation='relu', input_dim=3))
model.add(layers.Dense(12, activation='relu'))
model.add(layers.Dense(1, activation='sigmoid'))


print("BEFORE LEARNING")
weights = [layer.weights for layer in model.layers]
print("MODEL")
print(model.predict(data))
print("NP PREDICT")
print(np_predict(data, weights))
print("NAIVE PREDICT")
print(naive_predict(data, weights))


model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
H = model.fit(data, label, epochs=200, batch_size=2)


print("AFTER LEARNING")
weights = [layer.weights for layer in model.layers]
print("MODEL")
print(model.predict(data))
print("NP PREDICT")
print(np_predict(data, weights))
print("NAIVE PREDICT")
print(naive_predict(data, weights))
