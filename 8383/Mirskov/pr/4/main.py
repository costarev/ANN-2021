import numpy as np
from tensorflow.keras.layers import Dense
from tensorflow.keras.models import Sequential
import math


def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def numpy_func(weights, train_data):
    layer1 = np.maximum(np.dot(train_data, weights[0].get_weights()[0]) + weights[0].get_weights()[1], 0.) 
    layer2 = np.maximum(np.dot(layer1, weights[1].get_weights()[0]) + weights[1].get_weights()[1], 0.) 
    layer3 = sigmoid(np.dot(layer2, weights[2].get_weights()[0]) + weights[2].get_weights()[1]) 
    return layer3

def naive_relu(x):
    assert len(x.shape) == 2
    x = x.copy()
    for i in range(x.shape[0]):
        for j in range(x.shape[1]):
                x[i, j] = max(x[i, j], 0)      
    return x 

def naive_sigmoid(x):
    assert len(x.shape) == 2
    x = x.copy()
    for i in range(x.shape[0]):
        for j in range(x.shape[1]):
                x[i, j] = 1 / (1 + math.exp(-x[i, j]))     
    return x 

def naive_add_matrix_and_vector(x, y):
    assert len(x.shape) == 2
    assert len(y.shape) == 1
    assert x.shape[1] == y.shape[0]
    x = x.copy()
    for i in range(x.shape[0]):
        for j in range(x.shape[1]):
                x[i, j] += y[j]
    return x

def naive_vector_dot(x, y):
    assert len(x.shape) == 2
    assert len(y.shape) == 2
    assert x.shape[1] == y.shape[0]
    z = np.zeros([x.shape[0], y.shape[1]])
    for i in range(x.shape[0]):
        for j in range(y.shape[1]):
            s = 0
            for k in range(x.shape[1]):
                s += x[i, k] * y[k, j]
            z[i, j] += s
    return z

def elementwise_func(weights, train_data):
    layer1 = naive_relu(naive_add_matrix_and_vector(naive_vector_dot(train_data, weights[0].get_weights()[0]),
     weights[0].get_weights()[1])) 
    layer2 = naive_relu(naive_add_matrix_and_vector(naive_vector_dot(layer1, weights[1].get_weights()[0]),
     weights[1].get_weights()[1])) 
    layer3 = naive_sigmoid(naive_add_matrix_and_vector(naive_vector_dot(layer2, weights[2].get_weights()[0]),
     weights[2].get_weights()[1])) 
    return layer3

def logic_operation(a, b, c):
    return int((a and not b) or (c ^ b))

train_data = np.array([[0,0,0], [0,0,1], [0,1,0], [0,1,1], [1,0,0], [1,0,1], [1,1,0], [1,1,1]])
train_labels = np.array([logic_operation(*i) for i in train_data])

model = Sequential()
model.add(Dense(10, activation='relu', input_shape=(3,)))
model.add(Dense(10, activation='relu'))
model.add(Dense(1, activation='sigmoid'))

print('numpy_func')
print(numpy_func(model.layers, train_data))
print('elementwise_func')
print(elementwise_func(model.layers, train_data))

model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
history = model.fit(train_data, train_labels, epochs=1000, verbose = False)

print('numpy_func')
print(numpy_func(model.layers, train_data))
print('elementwise_func')
print(elementwise_func(model.layers, train_data))