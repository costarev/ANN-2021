import os

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import numpy as np
from keras import models
from keras import layers
import math


def numpy_f(layers, input):
    res = np.zeros((input.shape[0]))
    for i in range(len(res)):
        output1 = np.maximum(np.dot(np.transpose(layers[0].get_weights()[0]), input[i]) + layers[0].get_weights()[1], 0)
        output2 = np.maximum(np.dot(np.transpose(layers[1].get_weights()[0]), output1) + layers[1].get_weights()[1], 0)
        res[i] = sigmoid(np.dot(np.transpose(layers[2].get_weights()[0]), output2) + layers[2].get_weights()[1])
    return np.reshape(res, (len(res), 1))


def naive_f(layers, input):
    res = np.zeros((input.shape[0]))
    for i in range(len(res)):
        output1 = naive_relu(naive_matrix_vector_dot(np.transpose(layers[0].get_weights()[0]), input[i])
                             + layers[0].get_weights()[1])
        output2 = naive_relu(naive_matrix_vector_dot(np.transpose(layers[1].get_weights()[0]), output1)
                             + layers[1].get_weights()[1])
        res[i] = sigmoid(naive_matrix_vector_dot(np.transpose(layers[2].get_weights()[0]), output2)
                         + layers[2].get_weights()[1])
    return np.reshape(res, (len(res), 1))


def sigmoid(x):
    return 1 / (1 + math.exp(-x))


def naive_relu(x):
    assert len(x.shape) == 1
    x = x.copy()
    for i in range(x.shape[0]):
        x[i] = max(x[i], 0)
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


train_data = np.genfromtxt('task4_train.csv', delimiter=';')
train_labels = np.fromfile('task4_labels.csv', sep=';')

model = models.Sequential()
model.add(layers.Dense(16, activation='relu', input_shape=(3,)))
model.add(layers.Dense(16, activation='relu'))
model.add(layers.Dense(1, activation='sigmoid'))

print("Необученная сеть:")
print(model.predict(train_data))
print("NumPy:")
print(numpy_f(model.layers, train_data))
print("Naive:")
print(naive_f(model.layers, train_data))

model.compile(optimizer='rmsprop', loss='binary_crossentropy', metrics=['accuracy'])
history = model.fit(train_data, train_labels, epochs=1000, verbose=False)

print("-----------------------")
print("Обученная сеть:")
print(model.predict(train_data))
print("NumPy:")
print(numpy_f(model.layers, train_data))
print("Naive:")
print(naive_f(model.layers, train_data))
