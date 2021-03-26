import pandas
import numpy as np
from tensorflow.keras.layers import Dense
from tensorflow.keras.models import Sequential

def func(x):
    return (x[0] or x[1]) and (x[1] != (not x[2]))



def relu(x):
    return np.maximum(x, 0)


def sigmoid(x):
    return 1 / (1 + np.exp(-x))


def sim(data, weights):
    f = [relu, relu, sigmoid]
    r = data.copy()
    for i in range(len(weights)):
        res = np.zeros((r.shape[0], weights[i][0].shape[1]))
        for j in range(r.shape[0]):
            for k in range(weights[i][0].shape[1]):
                s = 0
                for m in range(r.shape[1]):
                    s += r[j][m] * weights[i][0][m][k]
                res[j][k] = f[i](s + weights[i][1][k])
        r = res
    return r


def numpy_sim(data, weights):
    f = [relu, relu, sigmoid]
    r = data.copy()
    for i in range(0, len(weights)):
        r = f[i](np.dot(r, weights[i][0]) + weights[i][1])
    return r


data = pandas.read_csv("number.csv", header=None).values.astype(int)
res = np.array([int(func(x)) for x in data])

model = Sequential()
model.add(Dense(16, activation='relu', input_shape=(3,)))
model.add(Dense(16, activation='relu'))
model.add(Dense(1, activation='sigmoid'))
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])


weights = [layer.get_weights() for layer in model.layers]
print("model predict")
print(str(model.predict(data)))
print("numpy")
print(str(numpy_sim(data, weights)))
print("no numpy")
print(str(sim(data, weights)))


H = model.fit(data, res, epochs=100, batch_size=1)

weights = [layer.get_weights() for layer in model.layers]
print("model predict")
print(str(model.predict(data)))
print("numpy")
print(str(numpy_sim(data, weights)))
print("no numpy")
print(str(sim(data, weights)))
