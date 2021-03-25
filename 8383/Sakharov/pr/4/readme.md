# Практическое задание 4


```python
import math
import pandas
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mclr
from tensorflow.keras.layers import Dense
from tensorflow.keras.models import Sequential
```

Функция из варианта 3


```python
def func(a, b, c):
    return (a and b) or c
```

Вспомогательные функции


```python
def relu(x):
    return np.maximum(x, 0)

def sigmoid(x):
    return 1 / (1 + np.exp(-x))
```

Реализация с помощью функций numpy


```python
def numpy_realization(layers, data):
    functions = [relu, relu, sigmoid]
    weights = [layer.get_weights() for layer in layers]
    layers = data.copy()
    for i, fun in enumerate(functions):
        layers = fun(layers @ weights[i][0] + weights[i][1])
    return layers
```

Собственная реализация


```python
def custom_realization(layers, data):
    functions = [relu, relu, sigmoid]
    weights = [layer.get_weights() for layer in layers]
    layers = data.copy()
    for i in range(len(weights)):
        nextLayers = np.zeros((layers.shape[0], weights[i][0].shape[1]))
        for j in range(layers.shape[0]):
            for k in range(weights[i][0].shape[1]):
                s = 0
                for m in range(layers.shape[1]):
                    s += layers[j][m] * weights[i][0][m][k]
                nextLayers[j][k] = functions[i](s + weights[i][1][k])
        layers = nextLayers
    return layers
```

Входные данные


```python
train_data = pandas.read_csv("data.csv", header=None).values.astype(int)
train_labels = np.array([int(func(x[0], x[1], x[2])) for x in train_data])
```

Модель сети


```python
model = Sequential()
model.add(Dense(16, activation='relu', input_shape=(3,)))
model.add(Dense(32, activation='relu'))
model.add(Dense(1, activation='sigmoid'))
```

Процесс обучения и проверки


```python
print("Необученная сеть:")
print(model.predict(train_data))
print("NumPy:")
print(numpy_realization(model.layers, train_data))
print("Naive:")
print(custom_realization(model.layers, train_data))

model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
history = model.fit(train_data, train_labels, epochs=1000, verbose=False)

print("Обученная сеть:")
print(model.predict(train_data))
print("NumPy:")
print(numpy_realization(model.layers, train_data))
print("Custom:")
print(custom_realization(model.layers, train_data))
```

    Необученная сеть:
    [[0.5       ]
     [0.5324743 ]
     [0.5139914 ]
     [0.5455347 ]
     [0.5465437 ]
     [0.53604746]
     [0.5153353 ]
     [0.51841754]]
    NumPy:
    [[0.5       ]
     [0.53247428]
     [0.5139914 ]
     [0.54553465]
     [0.54654369]
     [0.53604747]
     [0.5153353 ]
     [0.51841756]]
    Naive:
    [[0.5       ]
     [0.53247428]
     [0.5139914 ]
     [0.54553465]
     [0.54654369]
     [0.53604747]
     [0.5153353 ]
     [0.51841756]]
    Обученная сеть:
    [[0.00233829]
     [0.9995432 ]
     [0.00150636]
     [0.9999943 ]
     [0.00214076]
     [0.9998101 ]
     [0.99806213]
     [1.        ]]
    NumPy:
    [[0.0023383 ]
     [0.99954317]
     [0.00150635]
     [0.99999423]
     [0.00214078]
     [0.99981015]
     [0.99806207]
     [0.99999999]]
    Custom:
    [[0.0023383 ]
     [0.99954317]
     [0.00150635]
     [0.99999423]
     [0.00214078]
     [0.99981015]
     [0.99806207]
     [0.99999999]]
    


```python

```
