#var2

n_samp = 500
n_subset = int(n_samp/10)
n_epochs = 50

import numpy as np

import matplotlib.pyplot as plt

import matplotlib.colors as mclr

import tensorflow as tf

import pandas

from tensorflow.keras.layers import Dense

from tensorflow.keras.models import Sequential

from tensorflow.keras.utils import to_categorical

from sklearn.preprocessing import LabelEncoder

dataframe = pandas.read_csv("iris.csv", header=None)

dataset = dataframe.values

X = dataset[:,0:4].astype(float)

Y = dataset[:,4]

encoder = LabelEncoder()

encoder.fit(Y)

encoded_Y = encoder.transform(Y)

dummy_y = to_categorical(encoded_Y)

model = Sequential()

model.add(Dense(16, activation='relu'))
model.add(Dense(16, activation='relu'))
model.add(Dense(3, activation='softmax'))

model.compile(optimizer='adam',loss='categorical_crossentropy', metrics=['accuracy'])
# rmsprop
# adam
H = model.fit(X, dummy_y, epochs=100, batch_size=8, validation_split=0.1)

loss = H.history['loss']
val_loss = H.history['val_loss']
acc = H.history['accuracy']
val_acc = H.history['val_accuracy']
epochs = range(1, len(loss) + 1)

#Построение графика точности

plt.plot(epochs, acc, 'bo', label='Training acc')
plt.plot(epochs, val_acc, 'b', label='Validation acc')
plt.title('Training and validation accuracy')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend()
plt.show()

#Построение графика ошибки

plt.clf()
plt.plot(epochs, loss, 'bo', label='Training loss')
plt.plot(epochs, val_loss, 'b', label='Validation loss')
plt.title('Training and validation loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.show()