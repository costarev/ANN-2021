import pandas
from tensorflow.keras.layers import Dense
from tensorflow.keras.models import Sequential
from sklearn.preprocessing import LabelEncoder
import matplotlib.pyplot as plt


def draw_plot(xrange, training_data, validation_data, label):
    plt.plot(xrange, training_data, 'b', label='Training {}'.format(label))
    plt.plot(xrange, validation_data, 'r', label='Validation {}'.format(label))
    plt.xlabel('Epoch')
    plt.ylabel(label)
    plt.legend()
    plt.show()
    plt.clf()


def draw_result(history, nepochs):
    epochs = range(1, nepochs + 1)
    draw_plot(epochs, history['loss'], history['val_loss'], 'loss')
    draw_plot(epochs, history['accuracy'], history['val_accuracy'], 'accuracy')


dataframe = pandas.read_csv("sonar.csv", header=None)
dataset = dataframe.values
X = dataset[:, 0:60].astype(float)
Y = dataset[:, 60]

encoder = LabelEncoder()
encoder.fit(Y)
encoded_Y = encoder.transform(Y)

model = Sequential()
model.add(Dense(60, input_dim=60, activation='relu'))
model.add(Dense(15, activation='relu'))
model.add(Dense(1, activation='sigmoid'))
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
h = model.fit(X, encoded_Y, epochs=100, batch_size=10, validation_split=0.1)

draw_result(h.history, 100)
