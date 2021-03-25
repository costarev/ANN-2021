import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense


def customReluFunc(x):
    return np.maximum(x, 0)


def customSigmoidFunc(x):
    return 1 / (1 + np.exp(-x))


def sol(a, b, c):
    return (a or b) and (b or c)


x = np.array([[0, 0, 0], [0, 0, 1], [0, 1, 0], [0, 1, 1], [1, 0, 0], [1, 0, 1], [1, 1, 0], [1, 1, 1]])
y = np.array([sol(i[0], i[1], i[2]) for i in x])


def naive_dot_vXv(x, y):
    assert len(x.shape) == 1
    assert len(y.shape) == 1
    assert x.shape[0] == y.shape[0]
    z = 0.
    for i in range(x.shape[0]):
        z += x[i] * y[i]
    return z


def nativeDot(y, x):
    z = np.zeros(x.shape[1])
    for i in range(x.shape[1]):
        z[i] = naive_dot_vXv(y, x[:, i])
    return z


def nativeAdd(x, y):
    assert len(x.shape) == 1
    assert x.shape == y.shape
    x = x.copy()
    for i in range(x.shape[0]):
        x[i] += y[i]
    return x


def numpyEmulation(W, B, input):
    x = input.copy()
    for i in range(len(W)):
        x = np.dot(x, W[i])
        x += B[i]
        x = customReluFunc(x) if i != range(len(W))[-1] else customSigmoidFunc(x)
    return x


def nativeEmulation(W, B, input):
    x = input.copy()
    for i in range(len(W)):
        x = np.array([nativeDot(el, W[i]) for el in x])
        x = np.array([nativeAdd(el, B[i]) for el in x])
        x = [customReluFunc(el) for el in x] if i != range(len(W))[-1] else [customSigmoidFunc(el) for el in x]
    return np.array(x)


def solution(input, model):
    Weigths = [layer.get_weights()[0] for layer in model.layers]
    Bias = [layer.get_weights()[1] for layer in model.layers]
    print(f'предсказание с помощью нейросети библиотеки:\n{model.predict(input)}')
    print(f'эмуляция numpy:\n{numpyEmulation(Weigths, Bias, input)}')
    print(f'нативная эмуляция сети:\n{nativeEmulation(Weigths, Bias, input)}')


model = Sequential()
model.add(Dense(32, activation='relu', input_dim=3))
model.add(Dense(8, activation='relu'))
model.add(Dense(1, activation='sigmoid'))
model.compile(optimizer='rmsprop', loss='binary_crossentropy', metrics=['accuracy'])
print('====Необученная модель====')
solution(x, model)
H = model.fit(x, y, epochs=100, batch_size=1, verbose=0)
print('====Обученная модель====')
solution(x, model)
