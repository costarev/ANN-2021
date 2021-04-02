import numpy as np
from tensorflow.keras import layers
from tensorflow.keras import models


def operation(ex):
    return (ex[0] ^ ex[1]) and (ex[1] ^ ex[2])


def relu(x):
    return np.maximum(x, 0)


def sigmoid(x):
    return 1 / (1 + np.exp(-x))


def naive_matrix_vector_dot(x, w):
    assert len(w.shape) == 2
    assert len(x.shape) == 1
    assert w.shape[0] == x.shape[0]
    z = np.zeros(w.shape[1])
    for i in range(w.shape[1]):
        for j in range(w.shape[0]):
            z[i] += x[j] * w[j, i]
    return z


def naive_add(x, y):
    assert x.shape == y.shape
    x = x.copy()
    for i in range(x.shape[0]):
       x[i] += y[i]
    return x


def naive_simulation(x, weights):
    activationFuncs = [relu for i in range(len(weights) - 1)]
    activationFuncs.append(sigmoid)
    x = x.copy()
    for i in range(len(weights)):
        x = np.asarray([naive_matrix_vector_dot(v, weights[i][0]) for v in x])
        x = np.asarray([naive_add(v, weights[i][1]) for v in x])
        output = []
        for j in range(len(x)):
            output.append([activationFuncs[i](v) for v in x[j]])
        x = np.asarray(output)
    return x


def np_simulation(x, weights):
    activationFuncs = [relu for i in range(len(weights) - 1)]
    activationFuncs.append(sigmoid)
    x = x.copy()
    for i in range(len(weights)):
        x = np.dot(x, np.asarray(weights[i][0]))
        x = x + np.asarray(weights[i][1])
        x = activationFuncs[i](x)
    return x


x = np.array([[0, 0, 0],
              [0, 0, 1],
              [0, 1, 0],
              [0, 1, 1],
              [1, 0, 0],
              [1, 0, 1],
              [1, 1, 0],
              [1, 1, 1]])
# x = np.asarray(data)
y = np.asarray([operation(el) for el in x])
print("x: ", x)
print("Correct result: ", y)

model = models.Sequential()
model.add(layers.Dense(6, activation='relu', input_shape=(3,)))
model.add(layers.Dense(5, activation='relu'))
model.add(layers.Dense(1, activation='sigmoid'))

print("Прогон без обучения:")
print("Через модель")
print(model.predict(x))
weights = [layer.weights for layer in model.layers]
print("Через поэлементные операции:")
print(naive_simulation(x, weights))
print("Через операции над тензорами:")
print(np_simulation(x, weights))

# Обучение
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
H = model.fit(x, y, epochs=70, batch_size=4)

# Прогон после обучения
print("Прогон после обучения:")
print("Через модель")
print(model.predict(x))
weights = [layer.weights for layer in model.layers]
print("Через поэлементные операции:")
print(naive_simulation(x, weights))
print("Через операции над тензорами:")
print(np_simulation(x, weights))
