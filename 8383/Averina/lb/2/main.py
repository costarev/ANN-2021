import pandas
import matplotlib.pyplot as plt

from tensorflow.keras.layers import Dense
from tensorflow.keras.models import Sequential
from sklearn.preprocessing import LabelEncoder

dataframe = pandas.read_csv("sonar.csv", header=None)
dataset = dataframe.values
X = dataset[:, 0:30].astype(float)
Y = dataset[:, 60]

encoder = LabelEncoder()
encoder.fit(Y)
encoded_Y = encoder.transform(Y)

model = Sequential()
model.add(Dense(60, input_dim=30, kernel_initializer='normal', activation='relu')) #kernel_initializer - установка начальных случ. весов
model.add(Dense(15, kernel_initializer='normal', activation='relu'))
model.add(Dense(1, kernel_initializer='normal', activation='sigmoid'))

model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
H = model.fit(X, encoded_Y, epochs=100, batch_size=10, validation_split=0.1)
history = H.history

loss = history['loss']
acc = history['accuracy']
val_loss = history['val_loss']
val_acc = history['val_accuracy']
epochs = range(1, len(loss) + 1)


plt.plot(epochs, loss, label='Training loss', linestyle='--', linewidth=2, color="darkmagenta")
plt.plot(epochs, val_loss, 'b', label='Validation loss', color="lawngreen")
plt.title('Training and validation loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.show()

plt.clf()
plt.plot(epochs, acc, label='Training acc', linestyle='--', linewidth=2, color="darkmagenta")
plt.plot(epochs, val_acc, 'b', label='Validation acc', color="lawngreen")
plt.title('Training and validation accuracy')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend()
plt.show()