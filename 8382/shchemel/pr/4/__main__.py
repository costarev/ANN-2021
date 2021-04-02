import numpy as np
from tensorflow.keras import models
from tensorflow.keras.layers import Dense


def function(a: bool, b: bool, c: bool) -> bool:
    return (a & b) | (a & c)


def relu(x: float) -> np.array:
    return np.maximum(x, 0)


def sigmoid(x: float) -> np.array:
    return 1 / (1 + np.exp(-x))


def naive_dot(x: np.array, y: np.array) -> np.array:
    ret_array = np.zeros([x.shape[0], y.shape[1]])
    for i in range(x.shape[0]):
        for j in range(y.shape[1]):
            ret_array[i, j] = sum([x[i, k] * y[k, j] for k in range(x.shape[1])])
    return ret_array


def naive_sum(x: np.array, y: np.array) -> np.array:
    ret_array = np.zeros([x.shape[0], x.shape[1]])
    for i in range(x.shape[0]):
        for j in range(x.shape[1]):
            ret_array[i, j] = x[i, j] + y[j]
    return ret_array


def naive_network(data: np.array, weights: np.array) -> np.array:
    return sigmoid(naive_sum(
        naive_dot(relu(naive_sum(naive_dot(np.asarray(data), np.asarray(weights[0][0])), np.asarray(weights[0][1]))),
                  np.asarray(weights[1][0])), np.asarray(weights[1][1])))


def numpy_network(data: np.array, weights: np.array) -> np.array:
    return sigmoid(
        relu(data @ np.asarray(weights[0][0]) + np.asarray(weights[0][1])) @ np.asarray(weights[1][0]) + np.asarray(
            weights[1][1]))


def create_model() -> models.Model:
    model = models.Sequential()

    model.add(Dense(20, activation="relu"))
    model.add(Dense(1, activation="sigmoid"))

    model.compile(optimizer="adam", loss="binary_crossentropy", metrics=["accuracy"])

    return model


def main():
    data = []
    for i in range(2):
        for j in range(2):
            for k in range(2):
                data.append([i, j, k])
    labels = [function(i[0], i[1], i[2]) for i in data]

    model = create_model()

    model.fit(data, labels, epochs=400)

    weights = [layer.weights for layer in model.layers]

    print("trained keras: ", model.predict(data), sep="\n", end="\n=======================================\n")
    print("trained naive: ", naive_network(data, weights), sep="\n", end="\n=======================================\n")
    print("trained numpy: ", numpy_network(data, weights), sep="\n", end="\n=======================================\n")


if __name__ == "__main__":
    main()
