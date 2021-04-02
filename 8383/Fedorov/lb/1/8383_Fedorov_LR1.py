# импорт модулей
import pandas
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.layers import Dense
from tensorflow.keras.models import Sequential
from tensorflow.keras.utils import to_categorical
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from tensorflow.keras.utils import plot_model

dataframe = pandas.read_csv("iris.csv", header=None)
dataset = dataframe.values
X = dataset[:,0:4].astype(float)
Y = dataset[:,4]

encoder = LabelEncoder()
encoder.fit(Y)
encoded_Y = encoder.transform(Y)        
dummy_y = to_categorical(encoded_Y) 


# Создание модели сети
def create_model(num_layers=2, num_neurons=4):
    model = Sequential()
    model.add(Dense(num_neurons, activation='relu',  input_shape=(4,)))
    for i in range(num_layers-2):
        model.add(Dense(num_neurons, activation='relu'))
    model.add(Dense(3, activation='softmax'))    
    return model


# графики потерь и точности при обучении и тестирования
def plot_model_loss_and_accuracy(history, figsize_=(10,5)):
    plt.figure(figsize=figsize_)
    train_loss = history.history['loss']
    test_loss = history.history['val_loss']   
    train_acc = history.history['acc']
    test_acc = history.history['val_acc']    
    epochs = range(1, len(train_loss) + 1)
     
    plt.subplot(121)
    plt.plot(epochs, train_loss, 'r--', label='Training loss')
    plt.plot(epochs, test_loss, 'b-', label='Testing loss')
    plt.title('Graphs of losses during training and testing')
    plt.xlabel('epochs')
    plt.ylabel('loss')
    plt.legend()    
    plt.subplot(122)
    plt.plot(epochs, train_acc, 'r-.', label='Training acc')
    plt.plot(epochs, test_acc, 'b-', label='Testing acc')
    plt.title('Graphs of accuracy during training and testing')
    plt.xlabel('epochs', fontsize=11, color='black')
    plt.ylabel('accuracy', fontsize=11, color='black')
    plt.legend()
    plt.grid(True)
    
    plt.show()


model =  create_model(num_layers = 3, num_neurons = 36)
 
# инициализация параметров обучения (оптимизатор, функцию потерь,  метрика мониторинга (точность))
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])


# разделение данных на обучающие и тестовые в соотношении 80 : 20
#part_x_train, x_test, part_y_train, y_test = train_test_split(X, dummy_y, test_size=.20, random_state=42)
test_size = 16
part_x_train = np.concatenate((X[0:50-test_size], X[50:100-test_size], X[100:150-test_size]))
part_y_train = np.concatenate((dummy_y[0:50-test_size], dummy_y[50:100-test_size], dummy_y[100:150-test_size]))
x_test = np.concatenate((X[50-test_size:50], X[100-test_size:100], X[150-test_size:150]))
y_test = np.concatenate((dummy_y[50-test_size:50], dummy_y[100-test_size:100], dummy_y[150-test_size:150]))



# обучения сети  
history = model.fit(part_x_train, part_y_train, 
                              epochs=75, 
                              batch_size=5, 
                              validation_data=(x_test, y_test),
                              verbose=0)

plot_model_loss_and_accuracy(history)
plot_model(model, to_file='model.png', show_shapes=True)
