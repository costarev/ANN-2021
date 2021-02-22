import pandas
import numpy
from tensorflow.keras.layers import Dense
from tensorflow.keras.models import Sequential
from sklearn.preprocessing import LabelEncoder
import matplotlib.pyplot as plt

epochs = 100

dataframe = pandas.read_csv("sonar.csv", header=None)
dataset = dataframe.values
X = dataset[:, 0:30].astype(float)
Y = dataset[:, 60]

encoder = LabelEncoder()
encoder.fit(Y)
encoded_Y = encoder.transform(Y)

model = Sequential()
model.add(Dense(60, activation='relu', input_dim=30))
model.add(Dense(15, activation='relu'))
model.add(Dense(1, activation='sigmoid'))

model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
res = model.fit(X, encoded_Y, epochs=epochs, batch_size=10, validation_split=0.1)

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
