import pandas
from tensorflow.keras.layers import Dense
from tensorflow.keras.models import Sequential
import numpy as np

data = pandas.read_csv("data.csv", header=None).values.astype(int)
f = open("response.txt", "w")

def myRelu(x):
    return np.maximum(x, 0)


def mySigmoid(x):
    return 1 / (1 + np.exp(-x))


activations = [myRelu, myRelu, mySigmoid]


def v2(a, b, c):
    return (bool((a or b)) ^ bool(not(b and c)))


def nativeF(data, weights):
    layerData = data[:]
    for i, activation in enumerate(activations):
        res = [[0 for idx in range(weights[i][0].shape[1])]
               for jdx in range(len(layerData))]
        for k in range(len(res)):
            for m in range(len(res[0])):
                summa = sum([layerData[k][n] * weights[i][0][n][m]
                             for n in range(len(layerData[0]))])
                res[k][m] = activation(summa + weights[i][1][m])
        layerData = res
    return layerData


def numpyF(data, weights):
    layerData = data[:]
    for i, activation in enumerate(activations):
        layerData = activation(layerData @ weights[i][0] + weights[i][1])
    return layerData


def modelRun(data, model, f):
    weights = [layer.get_weights() for layer in model.layers]
    resModelPr = model.predict(data)
    resNative = nativeF(data, weights)
    resNumpy = numpyF(data, weights)
    parsedResModelPr = ("\n").join(
        [f'{i}: {num}' for i, arr in enumerate(resModelPr) for num in arr])
    parsedResNative = ("\n").join(
        [f'{i}: {num}' for i, arr in enumerate(resNative) for num in arr])
    parsedResNumpy = ("\n").join(
        [f'{i}: {num}' for i, arr in enumerate(resNumpy) for num in arr])
    f.write(f"\nmodel.predict\n{parsedResModelPr}")
    f.write(f"\nnative\n{parsedResNative}")
    f.write(f"\nnumpy\n{parsedResNumpy}")


answer = np.array([int(v2(x[0], x[1], x[2])) for x in data])

model = Sequential()
model.add(Dense(32, activation='relu', input_shape=(3,)))
model.add(Dense(16, activation='relu'))
model.add(Dense(1, activation='sigmoid'))
model.compile(optimizer='adam', loss='binary_crossentropy',
              metrics=['accuracy'])


f.write("Training\n")
modelRun(data, model, f)
model.fit(data, answer, epochs=100, batch_size=1)
f.write("\nValidation\n")
modelRun(data, model, f)
f.close()
