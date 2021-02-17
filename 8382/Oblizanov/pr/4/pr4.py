# v7

import pandas
from tensorflow.keras.layers import Dense
from tensorflow.keras.models import Sequential
import numpy as np


def relu(x):
    return np.maximum(x, 0)


def sigmoid(x):
    return 1 / (1 + np.exp(-x))


def expression(a, b, c):
    return (a or b) and (b != (not c))


def simulation(data, weights):
    afunc = [relu, relu, sigmoid]
    output = data.copy()
    for i in range(len(weights)):
        res = np.zeros((output.shape[0], weights[i][0].shape[1]))
        for j in range(output.shape[0]):
            for k in range(weights[i][0].shape[1]):
                s = 0
                for m in range(output.shape[1]):
                    s += output[j][m] * weights[i][0][m][k]
                res[j][k] = afunc[i](s + weights[i][1][k])
        output = res
    return output


def numpy_simulation(data, weights):
    afunc = [relu, relu, sigmoid]
    output = data.copy()
    for i in range(0, len(weights)):
        output = afunc[i](np.dot(output, weights[i][0]) + weights[i][1])
    return output


def model_prediction(data, model, f):
    weights = [layer.get_weights() for layer in model.layers]
    f.write("\nMODEL PREDICT\n")
    f.write(str(model.predict(data)))
    f.write("\nNUMPY SIMULATION\n")
    f.write(str(numpy_simulation(data, weights)))
    f.write("\nSIMULATION\n")
    f.write(str(simulation(data, weights)))


data = pandas.read_csv("data.csv", header=None).values.astype(int)
answer = np.array([int(expression(x[0], x[1], x[2])) for x in data])

model = Sequential()
model.add(Dense(16, activation='relu', input_shape=(3,)))
model.add(Dense(64, activation='relu'))
model.add(Dense(1, activation='sigmoid'))
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
f = open("output.txt", "w")
f.write("NO TRAINING\n")
model_prediction(data, model, f)
model.fit(data, answer, epochs=100, batch_size=1)
f.write("\n\nAFTER 100 EPOCHS\n")
model_prediction(data, model, f)
f.close()
