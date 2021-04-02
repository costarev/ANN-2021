import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense


def and_or_val(a, b, c):
    return (a and b) or c


def numpy_relu(x):
    return np.maximum(x, 0)


def numpy_sigmoid(x):
    return 1 / (1 + np.exp(-x))


def naive_dot(x, y):
    assert len(x.shape) == 1
    assert len(y.shape) == 1
    assert x.shape[0] == y.shape[0]
    z = 0.
    for i in range(x.shape[0]):
        z += x[i] * y[i]
    return z


def naive_matrix_dot(y, x):
    z = np.zeros(x.shape[1])
    for i in range(x.shape[1]):
        z[i] = naive_dot(y, x[:, i])
    return z


def naive_add(x, y):
    assert len(x.shape) == 1
    assert x.shape == y.shape
    x = x.copy()
    for i in range(x.shape[0]):
        x[i] += y[i]
    return x


def numpy_model(W, B, input):
    x = input.copy()
    for i in range(len(W)):
        x = np.dot(x, W[i])
        x += B[i]
        x = numpy_relu(x) if i != range(len(W))[-1] else numpy_sigmoid(x)
    return x


def naive_model(W, B, input):
    x = input.copy()
    for i in range(len(W)):
        x = np.array([naive_matrix_dot(v, W[i]) for v in x])
        x = np.array([naive_add(v, B[i]) for v in x])
        x = [numpy_relu(v) for v in x] if i != range(len(W))[-1] else [numpy_sigmoid(v) for v in x]
    return np.array(x)


def results(input, model):
    W = [layer.get_weights()[0] for layer in model.layers]
    B = [layer.get_weights()[1] for layer in model.layers]
    print('Predict\n', model.predict(input))
    print('NumPy\n', numpy_model(W, B, input))
    print('Regular\n', naive_model(W, B, input))


x = np.array([[0, 0, 0],
              [0, 0, 1],
              [0, 1, 0],
              [0, 1, 1],
              [1, 0, 0],
              [1, 0, 1],
              [1, 1, 0],
              [1, 1, 1]])
y = np.array([and_or_val(abc[0], abc[1], abc[2]) for abc in x])

print(y)
for k in range(8):
    print('(', x[k][0], 'and', x[k][1], ') or', x[k][2], '=', y[k])

model = Sequential()
model.add(Dense(8, activation='relu', input_dim=3))
model.add(Dense(16, activation='relu'))
model.add(Dense(1, activation='sigmoid'))
model.compile(optimizer='rmsprop', loss='binary_crossentropy', metrics=['accuracy'])

print('\nНеобученная модель\n')
results(x, model)

print('\nОбученная модель\n')
model.fit(x, y, epochs=100, batch_size=1, verbose=0)
results(x, model)