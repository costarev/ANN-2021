import pandas
import numpy
from tensorflow.keras.layers import Dense
from tensorflow.keras.models import Sequential
from tensorflow.keras.utils import to_categorical
from sklearn.preprocessing import LabelEncoder
import matplotlib.pyplot as plt

epochs = 100

dataframe = pandas.read_csv("iris.csv", header=None)
dataset = dataframe.values
X = dataset[:, 0:4].astype(float)
Y = dataset[:, 4]
encoder = LabelEncoder()
encoder.fit(Y)
encoded_Y = encoder.transform(Y)
dummy_y = to_categorical(encoded_Y)
model = Sequential()
model.add(Dense(4, activation='relu'))
model.add(Dense(192, activation='relu'))
model.add(Dense(192, activation='relu'))
model.add(Dense(3, activation='softmax'))
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
res = model.fit(X, dummy_y, epochs=epochs, batch_size=20, validation_split=0.1)

loss_history = numpy.array(res.history["loss"])
val_loss_history = numpy.array(res.history["val_loss"])
accuracy_history = numpy.array(res.history["accuracy"])
val_accuracy_history = numpy.array(res.history["val_accuracy"])

# graph maker
history = [loss_history, accuracy_history, val_loss_history, val_accuracy_history]
titles = ["Loss", "Accuracy", "Val Loss", "Val Accuracy"]
ylables = ["loss", "accur"]

for i in range(4):
    plt.subplot(2, 2, i + 1)
    plt.title(titles[i])
    plt.xlabel("epoch")
    plt.ylabel(ylables[i % 2])
    axes = plt.gca()
    if i % 2:
        axes.set_ylim([0, 1.1])
    else:
        axes.set_ylim([0, 3])
    plt.grid()
    plt.plot([i for i in range(epochs)], history[i], color="blue")
plt.show()
