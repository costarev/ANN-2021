import pandas

from tensorflow.keras.layers import Dense
from tensorflow.keras.models import Sequential
from tensorflow.keras.utils import to_categorical
from sklearn.preprocessing import LabelEncoder
import matplotlib.pyplot as plt

dataframe = pandas.read_csv("iris.csv", header=None)
dataset = dataframe.values
X = dataset[:, 0:4].astype(float)
Y = dataset[:, 4]


encoder = LabelEncoder()
encoder.fit(Y)
encoded_Y = encoder.transform(Y)
dummy_y = to_categorical(encoded_Y)

model = Sequential()
model.add(Dense(60, activation="relu"))
model.add(Dense(60, activation="relu"))
model.add(Dense(3, activation="softmax"))
model.compile(optimizer='adam', loss='categorical_crossentropy',
              metrics=['accuracy'])
result = model.fit(X, dummy_y, epochs=30,
                   batch_size=10, validation_split=0.1)

history = result.history
loss = history['loss']
val_loss = history['val_loss']
accuracy = history['accuracy']
val_accuracy = history['val_accuracy']
epochs = range(1, len(loss) + 1)

fig, axs = plt.subplots(2, 2)
fig.tight_layout(pad=3)

axs[0, 0].plot(epochs, val_loss, 'm', label='Validation loss')
axs[0, 0].set_title('Validation loss')
axs.flat[0].set(xlabel='', ylabel='Loss')


axs[0, 1].plot(epochs, loss, 'b', label='Training loss')
axs[0, 1].set_title('Training loss')
axs.flat[1].set(xlabel='', ylabel='')

axs[1, 0].plot(epochs, val_accuracy, 'm', label='Validation accuracy')
axs[1, 0].set_title('Validation accuracy')
axs.flat[2].set(xlabel='Epochs', ylabel='Accuracy')

axs[1, 1].plot(epochs, val_accuracy, 'b', label='Training accuracy')
axs[1, 1].set_title('Training accuracy')
axs.flat[3].set(xlabel='Epochs', ylabel='')

plt.show()
plt.clf()
