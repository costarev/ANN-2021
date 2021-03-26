import pandas
from tensorflow.keras.layers import Dense, Input
from tensorflow.keras.models import Sequential
from tensorflow.keras.utils import to_categorical
from sklearn.preprocessing import LabelEncoder
import matplotlib.pyplot as plt
import matplotlib.colors as mclr

def draw_plot(xrange, training_data, validation_data, label):
    plt.plot(xrange, training_data, 'b', label='Training {}'.format(label))
    plt.plot(xrange, validation_data, 'r', label='Validation {}'.format(label))
    plt.xlabel('Epoch')
    plt.ylabel(label)
    plt.legend()
    plt.show()
    plt.clf()

def draw_result(history, nepochs):
    loss = history['loss']
    val_loss = history['val_loss']
    accuracy = history['accuracy']
    val_accuracy = history['val_accuracy']
    epochs = range(1, nepochs + 1)
    draw_plot(epochs, history['loss'], history['val_loss'], 'loss')
    draw_plot(epochs, history['accuracy'], history['val_accuracy'], 'accuracy')


dataframe = pandas.read_csv("iris.csv", header=None)
dataset = dataframe.values
X = dataset[:, 0:4].astype(float)
Y = dataset[:, 4]

encoder = LabelEncoder()
encoder.fit(Y)
encoded_Y = encoder.transform(Y)
dummy_y = to_categorical(encoded_Y)

nepochs = 70

model = Sequential()
model.add(Input(shape=(4, )))
model.add(Dense(30, activation='relu'))
model.add(Dense(30, activation='relu'))
model.add(Dense(3, activation='softmax'))
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
h = model.fit(X, dummy_y, epochs=nepochs, batch_size=10, validation_split=0.1)

model.summary()

draw_result(h.history, nepochs)
