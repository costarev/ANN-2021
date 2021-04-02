import pandas
from tensorflow.keras.layers import Dense
from tensorflow.keras.models import Sequential
from sklearn.preprocessing import LabelEncoder
import matplotlib.pyplot as plt
import numpy as np

dataframe = pandas.read_csv("sonar.csv", header=None)
dataset = dataframe.values
np.random.shuffle(dataset)
X = dataset[:, 0:60].astype(float)
Y = dataset[:, 60]
encoder = LabelEncoder()
encoder.fit(Y)
encoded_Y = encoder.transform(Y)
model = Sequential()
model.add(Dense(60, input_dim=60, kernel_initializer='normal', activation='relu'))
#model.add(Dense(15, input_dim=60, kernel_initializer='normal', activation='relu'))
model.add(Dense(1, kernel_initializer='normal', activation='sigmoid'))
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
history = model.fit(X, encoded_Y, epochs=100, batch_size=10, validation_split=0.1)
plt.figure(1)
plt.plot(history.history['acc'], 'b', markersize=2)
plt.plot(history.history['val_acc'], 'r', markersize=2)
plt.title('Model accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend(['Train', 'Test'], loc='upper left')
plt.figure(2)
plt.plot(history.history['loss'], 'b', markersize=2)
plt.plot(history.history['val_loss'], 'r', markersize=2)
plt.title('Model loss')
plt.xlabel('Epoch')
plt.ylabel('loss')
plt.legend(['Train', 'Test'], loc='upper left')
plt.show()
