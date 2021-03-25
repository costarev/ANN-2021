import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense


def sigmoid(x):
    return 1 / (1 + np.exp(-x))


def relu(x):
    return np.maximum(x, 0)


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


def boolean_expression(x):
    return float(bool(x[0] and x[2] and x[1]) ^ bool(x[0] or not x[1]))


def add_vector_to_matrix(x, y):
    assert len(x.shape) == 2
    assert len(y.shape) == 1
    res = x.copy()
    for i in range(x.shape[0]):
        for j in range(x.shape[1]):
            res[i, j] = x[i, j] + y[j]
    return res


def native_solution(input_data, W, B):
    for i in range(len(W) - 1):
        input_data = relu(add_vector_to_matrix(matrices_dot(input_data, W[i]), B[i]))
    input_data = sigmoid(add_vector_to_matrix(matrices_dot(input_data, W[len(W) - 1]), B[len(W) - 1]))
    return input_data


def np_solution(input_data, W, B):
    for i in range(len(W) - 1):
        input_data = relu(np.dot(input_data, W[i]) + B[i])
    input_data = sigmoid(np.dot(input_data, W[len(W) - 1]) + B[len(W) - 1])
    return input_data


def all_solutions(input, model):
    W = [layer.get_weights()[0] for layer in model.layers]
    B = [layer.get_weights()[1] for layer in model.layers]

    print(f'Prediction:\n{model.predict(input)}')
    print(f'Numpy prediction:\n{np_solution(input, W, B)}')
    print(f'Native prediction:\n{native_solution(input, W, B)}')


train_data = np.genfromtxt('data.csv', delimiter=';')
train_label = np.asarray([boolean_expression(x) for x in train_data])
print(train_label)

model = Sequential()
model.add(Dense(16, activation='relu', input_dim=3))
model.add(Dense(16, activation='relu'))
model.add(Dense(1, activation='sigmoid'))
model.compile(optimizer='rmsprop', loss='binary_crossentropy', metrics=['accuracy'])
print('Untrained model\n')
all_solutions(train_data, model)

H = model.fit(train_data, train_label, epochs=100, batch_size=1, verbose=0)
print('Trained model\n')
all_solutions(train_data, model)
