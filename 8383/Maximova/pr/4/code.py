import pandas
import numpy as np
from tensorflow.keras.layers import Dense
from tensorflow.keras.models import Sequential

def build_model(sizeLayers):
    model = Sequential()
    model.add(Dense(sizeLayers[1], input_dim=sizeLayers[0], activation='relu'))
    model.add(Dense(sizeLayers[2], activation='relu'))
    model.add(Dense(sizeLayers[3], activation='sigmoid'))
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    return model

def getWeightsBias(model):
    weight_bias = model.get_weights()                                       # получение весов и смещений
    weights = [weight_bias[i] for i in range(0, len(weight_bias), 2)]
    bias = [weight_bias[i] for i in range(1, len(weight_bias), 2)]
    return weights, bias

def f1(weights, bias, input_layer, sizeLayers, index_layer):
    new_layer = np.zeros(sizeLayers[index_layer + 1])                       # для значений нейронов след слоя
    if(index_layer == len(weights)-1):                                      # sigmoid = 1/1+e^(-x)
        dot_ = 0
        for j in range(0, len(weights[index_layer])):
            dot_ += weights[index_layer][j] * input_layer[j]
        return 1 / (1 + np.exp(-(dot_ + bias[index_layer])))
    else:                                                                   # relu = max(0, x)
        for i in range(0, len(new_layer)):                                  # по нейроннам нового слоя
            dot_ = 0
            for j in range(0, len(weights[index_layer])):
                dot_ += weights[index_layer][j][i] * input_layer[j]
            new_layer[i] = max(dot_ + bias[index_layer][i], 0)
        return f1(weights, bias, new_layer, sizeLayers, index_layer + 1)

def f2(weights, bias, input_layer, sizeLayers, index_layer):
    if(index_layer == len(weights)-1):
        return 1 / (1 + np.exp(-(np.dot(input_layer, weights[index_layer]) + bias[index_layer])))
    else:
        return f2(weights, bias, np.maximum(np.dot(input_layer, weights[index_layer]) + bias[index_layer], 0),
                  sizeLayers, index_layer + 1)

def start(weights, bias, train_data, train_targets, sizeLayers, case):      # прогон датасета

    if(case):
        print("Прогон датасета через необученную модель и 2 реализованные функции:\n")
    else:
        print("Прогон датасета через обученную модель и 2 реализованные функции:\n")

    ins_res = model.predict(train_data)
    index = 0
    for obj in train_data:
        print("Дано: (", obj[0], "xor", obj[1], ") and (", obj[1], "xor", obj[2], ") = ", train_targets[index])
        print("Предсказание нейронной сети:", ins_res[index][0])
        print("Предсказание function1 (поэлементные операции):", f1(weights, bias, obj, sizeLayers, 0))
        print("Предсказание function2 (с использованием NumPy):", f2(weights, bias, obj, sizeLayers, 0))
        print("\n")
        index += 1

dataframe = pandas.read_csv("dataset.csv", header=None)
dataset = dataframe.values
train_data = dataset[:, 0:3]
train_targets = dataset[:, 3]
sizeLayers = np.array([3, 32, 16, 1])                                       # входной - скрытый - скрытый - выходной

model = build_model(sizeLayers)                                             # инициализация модели
weights_before, bias_before = getWeightsBias(model)
start(weights_before, bias_before, train_data, train_targets, sizeLayers, 1)

model.fit(train_data, train_targets, epochs=100, batch_size=1)
weights_after, bias_after = getWeightsBias(model)
start(weights_after, bias_after, train_data, train_targets, sizeLayers, 0)
