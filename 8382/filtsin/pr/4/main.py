import numpy as np
from tensorflow.keras import models
from tensorflow.keras.layers import Dense


def fn(a, b, c):
    return (a ^ b) & (b ^ c)


def relu(x):
    return np.maximum(x, 0)


def sigmoid(x):
    return 1 / (1 + np.exp(-x))


def mydot(x, y):
    r = np.zeros([x.shape[0], y.shape[1]])
    for i in range(x.shape[0]):
        for j in range(y.shape[1]):
            s = 0
            for k in range(x.shape[1]):
                s += x[i, k] * y[k, j]
            r[i, j] = s
    return r


def mysum(x, y):
    r = np.zeros([x.shape[0], x.shape[1]])
    for i in range(x.shape[0]):
        for j in range(x.shape[1]):
            r[i, j] = x[i, j] + y[j]
    return r


def tensor(x, w):
    return sigmoid(mysum(mydot(relu(mysum(mydot(np.asarray(x), np.asarray(w[0][0])), np.asarray(w[0][1]))), np.asarray(w[1][0])), np.asarray(w[1][1])))


def numpy(x, w):
    return sigmoid(relu(x@np.asarray(w[0][0]) + np.asarray(w[0][1]))@np.asarray(w[1][0]) + np.asarray(w[1][1]))


x = [[0, 0, 0], [0, 0, 1], [0, 1, 0], [0, 1, 1], [1, 0, 0], [1, 0, 1], [1, 1, 0], [1, 1, 1]]
y = [fn(i[0], i[1], i[2]) for i in x]


model = models.Sequential()
model.add(Dense(10, activation='relu', input_shape=(3,)))
model.add(Dense(1, activation='sigmoid'))
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
h = model.fit(x, y, epochs=800)

weights = [layer.weights for layer in model.layers]

print("keras: ", model.predict(x), sep='\n', end='\n\n')
print("numpy: ", numpy(x, weights), sep='\n', end='\n\n')
print("naive: ", tensor(x, weights), sep='\n', end='\n\n')