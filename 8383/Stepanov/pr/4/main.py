import numpy as np
from tensorflow.keras import layers
from tensorflow.keras import models

def f1 (weigts, data):

    layerPrevious = data

    for index in range(weigts.shape[0] // 2):   # цикл проходится по каждому из слоев

        layerCurr = np.zeros(weigts[index*2].shape[1])  # создается пустой слой

        for i in range(weigts[index*2].shape[1]):   # цикл проходится по индексам нейронов нынешнего слоя

            for j in range(weigts[index*2].shape[0]):   # цикл проходится по индексам нейронов предыдущего слоя
                layerCurr[i] += layerPrevious[j] * weigts[index*2][j][i]    # сумма весов и значений предыдущего нейрона
            layerCurr[i] += weigts[index*2+1][i]    # прибаление нейрона смещения

            if (index+1 != weigts.shape[0] // 2): # для послднего используем функцию активации sigmoid для остальных relu
                layerCurr[i] = max(layerCurr[i], 0) # relu
            else:
                layerCurr[i] = 1.0/(1.0 + np.exp(-layerCurr[i]))  # sigmoid

        layerPrevious = layerCurr

    return layerPrevious


def f2(weigts, data):

    layerPrevious = data

    for i in range(weigts.shape[0] // 2):   # цикл проходится по каждому из слоев

        layerCurr = np.dot(layerPrevious, weigts[i*2]) + weigts[i*2 + 1]    # сумма весов, значений предыдущего нейрона
                                                                            # и нейронов смещения
        if (i+1 != weigts.shape[0] // 2):   # для послднего используем функцию активации sigmoid для остальных relu
            layerCurr = np.maximum(layerCurr, 0)    # relu
        else:
            layerCurr = 1.0/(1.0 + np.exp(-layerCurr))   # sigmoid

        layerPrevious = layerCurr

    return layerPrevious


train_data = np.asarray([   [0, 0, 0],
                            [0, 0, 1],
                            [0, 1, 0],
                            [0, 1, 1],
                            [1, 0, 0],
                            [1, 0, 1],
                            [1, 1, 0],
                            [1, 1, 1]])

# (a xor b) and (b xor c)
train_label = np.asarray([0,0,1,0,0,1,0,0])

model = models.Sequential()
model.add(layers.Dense(32, activation='relu', input_dim=3))
model.add(layers.Dense(32, activation='relu'))
model.add(layers.Dense(1, activation= 'sigmoid'))

model.compile(optimizer='rmsprop', loss='binary_crossentropy', metrics=['accuracy'])

weights = model.get_weights()
for i in train_data:
    print(i, " = ", (i[0] ^ i[1])&(i[1] ^ i[2]))
    print("\tmodel = ", round(model.predict(np.asarray([i]))[0][0], 6))
    print("\tf1    = ", round(f1(np.asarray(weights), i)[0], 6))
    print("\tf2    = ", round(f2(np.asarray(weights), i)[0], 6))

model.fit(train_data, train_label, epochs=90, batch_size=1)

weights = model.get_weights()
for i in train_data:
    print(i, " = ", (i[0] ^ i[1])&(i[1] ^ i[2]))
    print("\tmodel = ", round(model.predict(np.asarray([i]))[0][0], 6))
    print("\tf1    = ", round(f1(np.asarray(weights), i)[0], 6))
    print("\tf2    = ", round(f2(np.asarray(weights), i)[0], 6))




