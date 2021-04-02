import pandas
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.layers import Dense
from tensorflow.keras.models import Sequential
from tensorflow.keras.utils import to_categorical
from sklearn.preprocessing import LabelEncoder
from tensorflow.keras.utils import plot_model
from pathlib import Path

def naive_relu(x):
    assert len(x.shape) == 2
    x = x.copy()
    for i in range(x.shape[0]):
        for j in range(x.shape[1]):
            x[i, j] = max(x[i, j], 0)
    
    return x    

def sigmoid(x):
    return 1/(1 + np.exp(x))     

def naive_vector_vector_dot(x, y):
    assert len(x.shape) == 1
    assert len(y.shape) == 1

    max_lenght = np.maximum(y.shape[0], x.shape[0])  
    
    z = 0
    for i in range(x.shape[0]):
        z += x[i % max_lenght] * y[i % max_lenght]

    return z        

def naive_matrix_matrix_dot(x, y):
    assert len(x.shape) == 2
    assert len(y.shape) == 2

    z = np.zeros((x.shape[0], y.shape[1]))
    for i in range(x.shape[0]):
        for j in range(y.shape[1]):
            z[i, j] = naive_vector_vector_dot(x[i, :], y[:, j]) 
    return z     

def numpy_predict(layers, input):
    k = len(layers) - 1
    res = input
    for i in range(k):
        res = np.maximum(np.dot(res, layers[i].get_weights()[0]) + layers[i].get_weights()[1], 0)
    res = sigmoid(np.dot(res, layers[k].get_weights()[0]) + layers[k].get_weights()[1])

    return 1-res

def naive_predict(layers, input):
    k = len(layers) - 1
    res = input
    for i in range(k):
        res = naive_relu(naive_matrix_matrix_dot(res, layers[i].get_weights()[0]) + layers[i].get_weights()[1])
    res = sigmoid(naive_matrix_matrix_dot(res, layers[k].get_weights()[0]) + layers[k].get_weights()[1])

    return 1-res    
# (a and b) or c

##Загрузка данных
path = Path("data.csv")
dataframe = pandas.read_csv(path.absolute(), header = None, sep=";")
dataset = dataframe.values
data_train = dataset[:, 0:3].astype(int)
data_labels = dataset[:,3].astype(int) 

model = Sequential()
model.add(Dense(8, activation="relu", input_shape=(3,)))
model.add(Dense(4, activation="relu"))
model.add(Dense(1, activation="sigmoid"))
model.compile(optimizer='rmsprop', loss='binary_crossentropy', metrics=['accuracy'])

print(model.predict(data_train))
print(numpy_predict(model.layers, data_train))
print(naive_predict(model.layers, data_train))

model.fit(data_train, data_labels, epochs=1000, verbose=0)

print("------------------------------------------------------------")
print(model.predict(data_train))
print(numpy_predict(model.layers, data_train))
print(naive_predict(model.layers, data_train))