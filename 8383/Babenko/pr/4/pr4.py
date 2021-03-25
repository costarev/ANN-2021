import pandas
from tensorflow.keras.layers import Dense
from tensorflow.keras.models import Sequential
import numpy as np

def sigmoid(x):
    return 1 / (1 + np.exp(-x))
def relu(x):
    return np.maximum(x, 0)
def expression(a, b, c):
    return (a or b) != (not (b and c))


def numpy_sim(data, weights):
    afunc = [relu, relu, sigmoid]
    output = data.copy()
    for i in range(0, len(weights)):
        output = afunc[i](np.dot(output, weights[i][0]) + weights[i][1])
    return output


def not_numpy_sim(data, weights):
    funcs = [relu, relu, sigmoid]
    result = data.copy()
    for i in range(len(weights)):
        res = np.zeros((result.shape[0], weights[i][0].shape[1]))
        for j in range(result.shape[0]):
            for k in range(weights[i][0].shape[1]):
                s = 0
                for m in range(result.shape[1]):
                    s += result[j][m] * weights[i][0][m][k]
                res[j][k] = funcs[i](s + weights[i][1][k])
        result = res
    return result


def model_prediction(file, data, model):
    weights = [layer.get_weights() for layer in model.layers]
    file.write("\nModel")
    file.write(str(model.predict(data)))
    file.write("\nNot Numpy")
    file.write(str(not_numpy_sim(data, weights)))
    file.write("\nNumpy")
    file.write(str(numpy_sim(data, weights)))


data = pandas.read_csv("input.csv", header=None).values.astype(int)
answer = np.array([int(expression(x[0], x[1], x[2])) for x in data])

model = Sequential()
model.add(Dense(20, activation='relu', input_shape=(3,)))
model.add(Dense(20, activation='relu'))
model.add(Dense(1, activation='sigmoid'))
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

fin = open("result.txt", "w")
fin.write("Before training\n")
model_prediction(fin, data, model)
model.fit(data, answer, epochs=50, batch_size=1)
fin.write("\nAfter training\n")
model_prediction(fin, data, model)
fin.close()
