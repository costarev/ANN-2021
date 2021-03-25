import pandas
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.layers import Dense
from tensorflow.keras.models import Sequential
from tensorflow.keras.utils import to_categorical
from sklearn.preprocessing import LabelEncoder
from tensorflow.keras.utils import plot_model


# графики потерь и точности при обучении и тестирования
def plot_model_loss_and_accuracy(history, figsize_=(10,5)):
    plt.figure(figsize=figsize_)
    train_loss = history.history['loss']
    train_acc = history.history['acc']
    epochs = range(1, len(train_loss) + 1)
     
    plt.subplot(121)
    plt.plot(epochs, train_loss, 'r--', label='Training loss')
    plt.title('Graphs of losses during training and testing')
    plt.xlabel('epochs')
    plt.ylabel('loss')
    plt.legend()    
    plt.subplot(122)
    plt.plot(epochs, train_acc, 'r-.', label='Training acc')
    plt.title('Graphs of accuracy during training and testing')
    plt.xlabel('epochs', fontsize=11, color='black')
    plt.ylabel('accuracy', fontsize=11, color='black')
    plt.legend()
    plt.grid(True)
    
    plt.show()

 
def naive_relu(x):
    assert len(x.shape) == 2 
    x = x.copy()
    for i in range(x.shape[0]):
        for j in range(x.shape[1]):
            x[i, j] = max(x[i, j], 0)      
    return x        
    

def sigmoid(x):
        return 1/(1 + np.exp(x))

def naive_vector_dot(x, y):
    assert len(x.shape) == 1                        # убедиться что x вектор
    assert len(y.shape) == 1                        # убедиться что y вектор 
    assert x.shape[0] == y.shape[0] 
    
    z = 0.
    for i in range(x.shape[0]):
        z += x[i] * y[i]
    return z

    
def naive_matrix_dot(x, y):
    assert len(x.shape) == 2                        # убедиться что x матрица
    assert len(y.shape) == 2                        # убедиться что y матрица 
    assert x.shape[1] == y.shape[0]
    
    z = np.zeros((x.shape[0], y.shape[1]))
    for i in range(x.shape[0]):
        for j in range(y.shape[1]):
            row_x = x[i, :]
            column_y = y[:, j]
            z[i, j] = naive_vector_dot(row_x, column_y)        
    return z        



def np_sim_model(layers, input):
    out_1 = np.maximum(np.dot(input, layers[0].get_weights()[0]) + layers[0].get_weights()[1], 0)
    out_2 = np.maximum(np.dot(out_1, layers[1].get_weights()[0]) + layers[1].get_weights()[1], 0)
    res = sigmoid(np.dot(out_2, layers[2].get_weights()[0]) + layers[2].get_weights()[1])
    
    return  1- res #reshape(res, len(res), 1))

def naive_sim_model(layers, input):    
    out_1 = naive_relu(naive_matrix_dot(input, layers[0].get_weights()[0]) + layers[0].get_weights()[1])
    out_2 = naive_relu(naive_matrix_dot(out_1, layers[1].get_weights()[0]) + layers[1].get_weights()[1])
    res = sigmoid(naive_matrix_dot(out_2, layers[2].get_weights()[0]) + layers[2].get_weights()[1])
    
    return 1- res


train_data = np.genfromtxt('train.csv', delimiter=';')
train_labels = np.fromfile('labels.csv', sep=';')


# создание модели ИНС
model = Sequential()
model.add(Dense(8, activation='relu', input_shape=(2,)))
model.add(Dense(8, activation='relu'))
model.add(Dense(1, activation='sigmoid'))
model.compile(optimizer='rmsprop', loss='binary_crossentropy', metrics=['accuracy'])


### Эксперимент
print('Untrained model:', model.predict(train_data))
print('Numpy fun', np_sim_model(model.layers, train_data))
print('Naive fun', naive_sim_model(model.layers, train_data))


history = model.fit(train_data, train_labels, epochs=450, verbose=False)
#plot_model_loss_and_accuracy(history)


print('Trained model:', model.predict(train_data))
print('Numpy fun', np_sim_model(model.layers, train_data))
print('Naive fun', naive_sim_model(model.layers, train_data))
