#!/usr/bin/env python
# coding: utf-8

# # Практическое задание 4

# In[2]:


import math
import pandas
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mclr
from tensorflow.keras.layers import Dense
from tensorflow.keras.models import Sequential


# Функция из варианта 3

# In[3]:


def func(a, b, c):
    return (a and b) or c


# Вспомогательные функции

# In[4]:


def relu(x):
    return np.maximum(x, 0)

def sigmoid(x):
    return 1 / (1 + np.exp(-x))


# Реализация с помощью функций numpy

# In[23]:


def numpy_realization(layers, data):
    functions = [relu, relu, sigmoid]
    weights = [layer.get_weights() for layer in layers]
    layers = data.copy()
    for i, fun in enumerate(functions):
        layers = fun(layers @ weights[i][0] + weights[i][1])
    return layers


# Собственная реализация

# In[37]:


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


# Входные данные

# In[25]:


train_data = pandas.read_csv("data.csv", header=None).values.astype(int)
train_labels = np.array([int(func(x[0], x[1], x[2])) for x in train_data])


# Модель сети

# In[26]:


model = Sequential()
model.add(Dense(16, activation='relu', input_shape=(3,)))
model.add(Dense(32, activation='relu'))
model.add(Dense(1, activation='sigmoid'))


# Процесс обучения и проверки

# In[38]:


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


# In[ ]:




