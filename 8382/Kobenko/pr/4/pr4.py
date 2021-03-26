import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense

### (a and c and b) xor (a or not b)
### output = act(dot(W, input) + B)

activation = {'sigmoid': lambda a: 1 / (1 + np.exp(-a)),
              'relu': lambda a: np.maximum(a, 0)}

def vector_dot(a, b):
    assert len(a.shape) == 1
    assert len(b.shape) == 1 
    assert a.shape[0] == b.shape[0]
    c = 0.
    for i in range(a.shape[0]):
        c += a[i] * b[i]
    return c

def matrix_dot(b, a):
    c = np.zeros(a.shape[1])
    for i in range(a.shape[1]):
        c[i] = vector_dot(b, a[:, i])
    return c

def matrix_add_vector(a, b):
    assert len(a.shape) == 1
    assert a.shape == b.shape   
    a = a.copy() 
    for i in range(a.shape[0]):
        a[i] += b[i]
    return a

def bool_operation(a, b, c):
    f = a and c and b
    ff = a or not b
    return f ^ ff

def np_res(W, B, input): # функция с использованием numpy
    a = input.copy()
    for i in range(len(W)):
        a = np.dot(a, W[i])
        a += B[i]
        a = activation['relu'](a) if i != range(len(W))[-1] else activation['sigmoid'](a)
    return a

def elem_res(W, B, input): # функция с использованием поэлементных операций
    a = input.copy()
    for i in range(len(W)):
        a = np.array([matrix_dot(el, W[i]) for el in a])
        a = np.array([matrix_add_vector(el, B[i]) for el in a])
        a = [activation['relu'](el) for el in a] if i != range(len(W))[-1] else [activation['sigmoid'](el) for el in a]
    return np.array(a)

def result(input, model):
    W = [layer.get_weights()[0] for layer in model.layers]
    B = [layer.get_weights()[1] for layer in model.layers]
    layer_names = [layer.name for layer in model.layers]
    print(f'predict:\n{model.predict(input)}')
    print(f'numpy:\n{np_res(W, B, input)}')
    print(f'elem:\n{elem_res(W, B, input)}')

a = np.array([[1, 0, 1], [1, 1, 1], [0, 1, 0], [1, 1, 0],
              [0, 0, 0], [0, 0, 1], [1, 0, 0], [0, 1, 1]])
b = np.array([bool_operation(i[0],i[1],i[2]) for i in a])

model = Sequential()
model.add(Dense(16, activation='relu', input_dim=3))
model.add(Dense(8, activation='relu'))
model.add(Dense(1, activation='sigmoid'))
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
print('Необученная модель\n')
result(a, model)
H = model.fit(a, b, epochs=100, batch_size=1, verbose=0)
print('\nОбученная модель\n')
result(a, model)