import numpy as np
from tensorflow.keras.layers import Dense
from tensorflow.keras.models import Sequential


def logic_func(a, b, c):
    return (a or b) and (b != (not c))


def relu(x):
    return np.maximum(x, 0)


def sigmoid(x):
    return 1 / (1 + np.exp(-x))


functions = [relu, relu, sigmoid]


def np_simulation(data, m_weights):
    layers = data.copy()
    for i, fun in enumerate(functions):
        layers = functions[i](np.dot(layers, m_weights[i][0]) + m_weights[i][1])
    return layers


def native_simulation(data, m_weights):
    layers = data.copy()
    for i in range(len(m_weights)):
        next_layers = np.zeros((layers.shape[0], m_weights[i][0].shape[1]))
        for j in range(layers.shape[0]):
            for k in range(m_weights[i][0].shape[1]):
                s = 0
                for m in range(layers.shape[1]):
                    s += layers[j][m] * m_weights[i][0][m][k]
                next_layers[j][k] = functions[i](s + m_weights[i][1][k])
        layers = next_layers
    return layers


train_data = np.loadtxt('input.txt', dtype=int, delimiter=' ')
train_labels = np.array([int(logic_func(x[0], x[1], x[2])) for x in train_data])

model = Sequential()
model.add(Dense(16, activation='relu', input_shape=(3,)))
model.add(Dense(16, activation='relu'))
model.add(Dense(1, activation='sigmoid'))

weights = [layer.get_weights() for layer in model.layers]
print("Модель до обученя:")
print(model.predict(train_data))
print("Функция с поэлементными операциями:")
print(native_simulation(train_data, weights))
print("Функция использующая numpy:")
print(np_simulation(train_data, weights))

model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
history = model.fit(train_data, train_labels, epochs=300, verbose=False)

print("Модель после обученя:")
print(model.predict(train_data))
print("Функция с поэлементными операциями:")
print(native_simulation(train_data, weights))
print("Функция использующая numpy:")
print(np_simulation(train_data, weights))
