
import pandas
from tensorflow.keras.layers import Dense
from tensorflow.keras.models import Sequential
import numpy as np
import math

# Var 4
def gt(a:bool,b:bool,c:bool) -> int:
    return int(((a | b) & (b | c)))

def get_data():
    data = []
    targets = []
    for i in range(2):
        for j in range(2):
            for k in range(2):
                data.append([i,j,k])
                targets.append(gt(i==1,j==1,k==1))
    return np.array(data), np.array(targets)

def get_model()->Sequential:
    model = Sequential()
    model.add(Dense(16, activation='relu', input_shape=(3,)))
    model.add(Dense(32, activation='relu'))
    model.add(Dense(1, activation='sigmoid'))
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    return model

def exec_numpy(layers,activations,data:np.ndarray):
    state = data.copy()
    for i in range(len(layers)):
        state = activations[i](np.dot(state,np.array(layers[i][0]))+layers[i][1])
    return state

def naive_relu(x):
    assert len(x.shape) == 2
    x = x.copy()
    for i in range(x.shape[0]):
        for j in range(x.shape[1]):
            x[i,j] = max(x[i, j], 0)
    return x

def naive_sigmoid(x):
    assert len(x.shape) == 2
    x = x.copy()
    for i in range(x.shape[0]):
        for j in range(x.shape[1]):
                x[i, j] = 1 / (1 + math.exp(-x[i, j]))
    return x


def naive_add(x, y):
    assert len(x.shape) == 2
    if x.shape != y.shape:
        assert y.shape[0] == x.shape[1]
        y = np.repeat(y[np.newaxis,:],x.shape[0],0)
    x = x.copy()
    for i in range(x.shape[0]):
        for j in range(x.shape[1]):
            x[i, j] += y[i, j]
    return x

def naive_vector_dot(x, y):
    assert len(x.shape) == 1
    assert len(y.shape) == 1
    assert x.shape[0] == y.shape[0]
    z = 0.
    for i in range(x.shape[0]):
        z += x[i] * y[i]
    return z

def naive_dot(x:np.ndarray,y:np.ndarray):
    assert x.shape[1] == y.shape[0]
    z = np.zeros([x.shape[0],y.shape[1]])
    for i in range(x.shape[0]):
        for j in range(y.shape[1]):
            z[i,j] = naive_vector_dot(x[i,:],y[:,j])
    return z

def exec_naive(layers,data:np.ndarray):
    activations = [naive_relu,naive_relu, naive_sigmoid]
    state = data.copy()
    for i in range(len(layers)):
        state = activations[i](naive_add(naive_dot(state, (layers[i][0])), layers[i][1]))
    return state


def predict(model:Sequential, data:np.ndarray):
    weights = [i.weights for i in model.layers]
    activations = [i.activation for i in model.layers]
    print("Execute model")
    print(model(data))
    print("Execute with numpy")
    print(exec_numpy(weights,activations,data))
    print("Execute with naive")
    print(exec_naive(weights, data))




if __name__=="__main__":
    model = get_model()
    data = get_data()

    print("BEFORE FIT:")
    predict(model,data[0])
    epochs = 100
    H = model.fit(data[0],data[1],epochs=epochs,batch_size=1,verbose=False)
    print()
    print(f'AFTER FIT({epochs} epochs)')
    predict(model,data[0])
    # print(model(data[0]) - trg)
