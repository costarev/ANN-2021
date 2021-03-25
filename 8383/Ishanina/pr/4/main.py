from tensorflow.keras.layers import Dense
from tensorflow.keras.models import Sequential
import numpy as np

def relu(x):
    return np.maximum(x, 0)

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def expression(a, b):
    return (a or b) and (a != (not b))

def tensor_analogue(data, weights):
    arr_func = [relu, relu, sigmoid]
    input_data = data.copy()
    for i in range(len(weights)):
        answer = np.zeros((input_data.shape[0], weights[i][0].shape[1]))
        for j in range(input_data.shape[0]):
            for k in range(weights[i][0].shape[1]):
                s = 0
                for m in range(input_data.shape[1]):
                    s += input_data[j][m] * weights[i][0][m][k]
                answer[j][k] = arr_func[i](s + weights[i][1][k])
        input_data = answer
    return input_data

def numpy_analogue(data, weights):
    arr_func = [relu, relu, sigmoid]
    input_data = data.copy()
    for i in range(0, len(weights)):
        input_data = arr_func[i](np.dot(input_data, weights[i][0]) + weights[i][1])
    return input_data

data = np.array([[0,0],[0,1],[1,0],[1,1]])
answer = np.array([int(expression(x[0], x[1])) for x in data])

model = Sequential()
model.add(Dense(10, activation='relu', input_shape=(2,)))
model.add(Dense(10, activation='relu'))
model.add(Dense(1, activation='sigmoid'))
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

print("До обучения")
weights = [layer.get_weights() for layer in model.layers]
print("Результат модели")
print(str(model.predict(data)))
print("Функция с numpy")
print(str(numpy_analogue(data, weights)))
print("Функция с тензорными операциями")
print(str(tensor_analogue(data, weights)))
model.fit(data, answer, epochs=70, batch_size=1)
print("После обучения")
weights = [layer.get_weights() for layer in model.layers]
print("Результат модели")
print(str(model.predict(data)))
print("Функция с numpy")
print(str(numpy_analogue(data, weights)))
print("Функция с тензорными операциями")
print(str(tensor_analogue(data, weights)))
