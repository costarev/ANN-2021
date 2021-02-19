import pandas
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.layers import Dense
from tensorflow.keras.models import Sequential
from tensorflow.keras.utils import to_categorical
from sklearn.preprocessing import LabelEncoder
from tensorflow.keras.utils import plot_model

# Х - входные данные, У-выходные, данные перемешиваются
dataframe = pandas.read_csv("sonar.csv", header=None)
dataset = dataframe.values
np.random.shuffle(dataset)

X = dataset[:,0:60].astype(float)
Y = dataset[:,60]


#Переход от R,M меток к категориальному вектору
encoder = LabelEncoder()
encoder.fit(Y)
encoded_Y = encoder.transform(Y)   #[1, 1, 1, 1, ... , 0 , 0, 0]


###### отбор тестовых данных
test_index = int(len(X)-len(X)*0.2)
test_data_x = X[test_index:]

X = X[:test_index]
test_index = int(len(encoded_Y)-len(encoded_Y)*0.2)
test_data_y = encoded_Y[test_index:]
encoded_Y = encoded_Y[:test_index]


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


def get_model():
    model = Sequential()
    model.add(Dense(60, input_dim=60, activation='relu'))
    model.add(Dense(15, activation='relu'))
    model.add(Dense(1, activation='sigmoid')) 
    model.compile(optimizer='adam',loss='binary_crossentropy', metrics=['accuracy'])
    return model

### Другой вариант обучения 
k = 4
num_valid_data = len(X) // k
valid_scores_loss = []
valid_scores_acc = []


#X = X[:, 0:50]
#test_data_x = test_data_x[:, 0:50]
#перекрестная проверка по К блокам
for fold in range(k):
    validation_data_x = X[num_valid_data * fold: num_valid_data * (fold + 1)]
    validation_data_y = encoded_Y[num_valid_data * fold: num_valid_data * (fold + 1)]

    train_data_x = np.concatenate((X[:num_valid_data * fold] , X[num_valid_data * (fold + 1):])) 
    train_data_y = np.concatenate((encoded_Y[:num_valid_data * fold] , encoded_Y[num_valid_data * (fold + 1):])) 
    
    model = get_model()
    history = model.fit(train_data_x, train_data_y, epochs=100, batch_size=10, verbose=0)
    results = model.evaluate(validation_data_x, validation_data_y, batch_size=10,verbose=0)
    valid_scores_loss.append(results[0])
    valid_scores_acc.append(results[1])
 
valid_score_acc = np.average(valid_scores_acc)
valid_score_loss = np.average(valid_scores_loss)
print("Loss: ", valid_score_loss)
print("Accuracy: ", valid_score_acc)

#обучение на всех данных, кроме контрольного набора
model = get_model()
history = model.fit(X, encoded_Y, epochs=100, batch_size=10, validation_data=(test_data_x, test_data_y),verbose=2)
#history = model.fit(X_, encoded_Y_, epochs=100, batch_size=10, validation_split=0.2, verbose=2)
test_score = model.evaluate(test_data_x,test_data_y, batch_size=10,verbose=0)
print(test_score)
plot_model_loss_and_accuracy(history)
plot_model(model, to_file='model.png', show_shapes=True)
