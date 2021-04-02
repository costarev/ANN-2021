import pandas
import matplotlib.pyplot as plt
import numpy as np
from tensorflow.keras.layers import Dense
from tensorflow.keras.models import Sequential
from tensorflow.keras.utils import to_categorical
from sklearn.preprocessing import LabelEncoder

#Загрузка данных
dataframe = pandas.read_csv("iris.csv", header=None)
dataset = dataframe.values

np.random.shuffle(dataset)

X = dataset[:,0:4].astype(float)
Y = dataset[:,4]

#Переход от текстовых меток к категориальному вектору
encoder = LabelEncoder()
encoder.fit(Y)
encoded_Y = encoder.transform(Y)
dummy_y = to_categorical(encoded_Y)

#Создание модели
model = Sequential()
model.add(Dense(16, activation='relu', input_shape=(4,)))
model.add(Dense(16, activation='relu'))
model.add(Dense(3, activation='softmax'))

#Инициализация параметров обучения
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

#Обучение сети
history = model.fit(X, dummy_y, epochs=75, batch_size=10, validation_split=0.1)

history_dict = history.history
#print(history_dict.keys())

#График ошибки
loss_values = history_dict['loss']
val_loss_values = history_dict['val_loss']
epochs = range(1, len(loss_values) +1)
plt.plot(epochs, loss_values, 'g', label='Training loss')
plt.plot(epochs, val_loss_values, 'y', label='Validation loss')
plt.title('Training and validation loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.show()

#График точности
plt.clf()
acc_values = history_dict['accuracy']
val_acc_values = history_dict['val_accuracy']
plt.plot(epochs, acc_values, 'r', label='Training accurate')
plt.plot(epochs, val_acc_values, 'b', label='Validation accurate')
plt.title('Training and validation accuracy')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend()
plt.show()


