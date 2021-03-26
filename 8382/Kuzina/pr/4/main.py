import numpy as np
from tensorflow.keras.layers import Dense
from tensorflow.keras.models import Sequential


def relu(x):
    return np.maximum(x, 0)


def sigmoid(x):
    return 1/(1 + np.exp(x))


def vectors_dot(x, y):
    assert len(x.shape) == 1
    assert len(y.shape) == 1
    assert x.shape[0] == y.shape[0]
    res = 0.0
    for i in range(x.shape[0]):
        res += x[i] * y[i]
    return res


def matrices_dot(x, y):
    assert len(x.shape) == 2
    assert len(y.shape) == 2
    assert x.shape[1] == y.shape[0]
    res = np.zeros((x.shape[0], y.shape[1]))
    for i in range(x.shape[0]):
        for j in range(y.shape[1]):
            res[i, j] = vectors_dot(x[i, :], y[:, j])
    return res


def numpy_predict(layers, input):
    k = len(layers) - 1
    res = input
    for i in range(k):
        res = relu(np.dot(res, layers[i].get_weights()[0]) + layers[i].get_weights()[1])
    res = sigmoid(np.dot(res, layers[k].get_weights()[0]) + layers[k].get_weights()[1])

    return 1-res


def naive_predict(layers, input):
    k = len(layers) - 1
    res = input
    for i in range(k):
        res = relu(matrices_dot(res, layers[i].get_weights()[0]) + layers[i].get_weights()[1])
    res = sigmoid(matrices_dot(res, layers[k].get_weights()[0]) + layers[k].get_weights()[1])

    return 1-res


def testing(model, entering):
    print(entering)
    print("\nРезультаты модели:")
    print(model.predict(data_train))
    print("Результаты операций над тензорами numpy:")
    print(numpy_predict(model.layers, data_train))
    print("Результаты поэлементных операций над тензорами:")
    print(naive_predict(model.layers, data_train))


try:
    #(a and b) or (a and c)
    data = np.genfromtxt('input.csv', dtype='int', delimiter=';')
except IndexError:
    print("Ошибка входных данных")

data_train = data[:, 0:3].astype(int)
data_labels = data[:,3].astype(int)

model = Sequential()
model.add(Dense(8, activation="relu", input_shape=(3,)))
model.add(Dense(5, activation="relu"))
model.add(Dense(1, activation="sigmoid"))
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

testing(model, "\nДо обучениия")

model.fit(data_train, data_labels, epochs=500, batch_size=5, verbose=0)

testing(model, "\nПосле обучения")
