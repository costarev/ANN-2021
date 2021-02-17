import pandas
import numpy as np
from matplotlib import pyplot
from tensorflow.keras.layers import Dense
from tensorflow.keras.models import Sequential
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.utils import plot_model
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split

from pathlib import Path

##Загрузка данных
path = Path("iris.csv")
dataframe = pandas.read_csv(path.absolute(), header = None)
dataset = dataframe.values
X = dataset[:, 0:4].astype(float)
Y = dataset[:,4]


##Создание категориального вектора 
encoder = LabelEncoder()
encoder.fit(Y)
encoded_Y = encoder.transform(Y)
dummy_Y = to_categorical(encoded_Y)

##Создание модели
model = Sequential()
model.add(Dense(32, activation="relu", input_shape=(4,)))
model.add(Dense(32, activation="relu"))
model.add(Dense(3, activation="softmax"))

##Установка параметров обучения
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

##Обучение модели
data = model.fit(X, dummy_Y, epochs = 75, batch_size = 10, validation_split = 0.1, verbose=0)  

##Печать модели
plot_model(model, to_file="model.png", show_shapes = True)

##Отрисовка графиков потерь и точности
loss = data.history['loss']
val_loss = data.history['val_loss']   
accuracy = data.history['accuracy']
val_accuracy = data.history['val_accuracy']    
epochs = range(0, 75)
pyplot.figure()

##График потерь
pyplot.subplot(2, 1, 1)
pyplot.plot(epochs, loss, 'm+', label='Training loss')
pyplot.plot(epochs, val_loss, 'y-', label='Validation loss')
pyplot.title('Chart of losses')
pyplot.xlabel('Epochs')
pyplot.ylabel('Losses')
pyplot.legend()    

##График точности
pyplot.subplot(2, 1, 2)
pyplot.plot(epochs, accuracy, 'm+', label='Training accuracy')
pyplot.plot(epochs, val_accuracy, 'y-', label='Validation accuracy')
pyplot.title('Chart of accuracy')
pyplot.xlabel('Epochs')
pyplot.ylabel('Accuracy')
pyplot.legend()

pyplot.tight_layout()
pyplot.show() 