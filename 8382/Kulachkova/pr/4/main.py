import numpy as np
from tensorflow.keras import layers
from tensorflow.keras import models


def operation(x):
    return (x[0] or x[1]) and (x[1] or x[2])


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


def naive_add(x, w):
    assert x.shape == w.shape
    x = x.copy()
    for i in range(x.shape[0]):
        x[i] += w[i]
    return x


def naive_simulation(x, w):
    n_layers = len(w)
    act_funcs = [relu, sigmoid]
    assert n_layers == len(act_funcs)
    x = x.copy()
    for i in range(n_layers):
        x = np.asarray([naive_matrix_vector_dot(v, w[i][0]) for v in x])
        x = np.asarray([naive_add(v, w[i][1]) for v in x])
        output = []
        for j in range(len(x)):
            output.append([act_funcs[i](v) for v in x[j]])
        x = np.asarray(output)
    return x


def np_simulation(x, w):
    n_layers = len(w)
    act_funcs = [relu, sigmoid]
    assert n_layers == len(act_funcs)
    x = x.copy()
    for i in range(n_layers):
        x = np.dot(x, np.asarray(w[i][0]))
        x = x + np.asarray(w[i][1])
        x = act_funcs[i](x)
    return x


def testing(model, x):
    print("ПРОГОН ЧЕРЕЗ МОДЕЛЬ")
    print(model.predict(x))
    weights = [layer.weights for layer in model.layers]
    print("ПОЭЛЕМЕНТНЫЕ ОПЕРАЦИИ")
    print(naive_simulation(x, weights))
    print("ОПЕРАЦИИ НАД ТЕНЗОРАМИ")
    print(np_simulation(x, weights))


# Генерация данных
data = []
for a in [0, 1]:
    for b in [0, 1]:
        for c in [0, 1]:
            data.append([a, b, c])
x = np.asarray(data)
y = np.asarray([operation(el) for el in x])
print("x: ", x)

# Инициализация модели
model = models.Sequential()
model.add(layers.Dense(6, activation='relu', input_shape=(3,)))
model.add(layers.Dense(1, activation='sigmoid'))

# Прогон без обучения
print("Прогон без обучения")
testing(model, x)

# Обучение
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
H = model.fit(x, y, epochs=180, batch_size=4)

# Прогон после обучения
print("Прогон после обучения (180 эпох)")
testing(model, x)
print("y: ", y)
