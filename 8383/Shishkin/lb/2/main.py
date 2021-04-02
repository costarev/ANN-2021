import pandas
import tensorflow as tf
from tensorflow.keras.layers import Dense
from tensorflow.keras.models import Sequential
from tensorflow.keras.utils import to_categorical
from sklearn.preprocessing import LabelEncoder

import matplotlib.pyplot as plt


def one_launch(epochss, batchSizee, validationSplitt):
    model = Sequential()
    model.add(Dense(60, input_dim=30, activation='relu'))
    model.add(Dense(15, activation='relu'))
    model.add(Dense(5, activation='relu'))
    model.add(Dense(1, activation='sigmoid'))
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    H = model.fit(X, encoded_Y, epochs=epochss, batch_size=batchSizee, validation_split=validationSplitt)

    loss = H.history['loss']
    val_loss = H.history['val_loss']
    acc = H.history['accuracy']
    val_acc = H.history['val_accuracy']
    epochs = range(1, len(loss) + 1)

    plt.plot(epochs, loss, 'bo', label='Training loss')
    plt.plot(epochs, val_loss, 'b', label='Validation loss')
    plt.title('Training and validation loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.show()

    plt.clf()
    plt.plot(epochs, acc, 'bo', label='Training acc')
    plt.plot(epochs, val_acc, 'b', label='Validation acc')
    plt.title('Training and validation accuracy')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.show()


def multiple_launches(num, epochss, batchSizee, validationSplitt):
    c = []
    for i in range(num):
        model = Sequential()
        model.add(Dense(60, input_dim=30, activation='relu'))
        model.add(Dense(15, activation='relu'))
        model.add(Dense(5, activation='relu'))
        model.add(Dense(1, activation='sigmoid'))
        model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
        H = model.fit(X, encoded_Y, epochs=epochss, batch_size=batchSizee, validation_split=validationSplitt)
        c.append(H)

    print('VAL_ACCURACY: ', end='')
    for val in c:
        print(str(round(((val.history['val_accuracy'])[epochss - 1]), 2)) + ', ', end='')

    print("")
    print('ACCURACY: ', end='')
    for val in c:
        print(str(round(((val.history['accuracy'])[epochss - 1]), 2)) + ', ', end='')

    print("")
    print('VAL_LOSS: ', end='')
    for val in c:
        print(str(round(((val.history['val_loss'])[epochss - 1]), 2)) + ', ', end='')

    print("")
    print('LOSS: ', end='')
    for val in c:
        print(str(round(((val.history['loss'])[epochss - 1]), 2)) + ', ', end='')


dataframe = pandas.read_csv("sonar.csv", header=None)
dataset = dataframe.values
X = dataset[:, 0:30].astype(float)
Y = dataset[:, 60]

encoder = LabelEncoder()
encoder.fit(Y)
encoded_Y = encoder.transform(Y)

one_launch(100, 10, 0.1)
# multiple_launches(20, 100, 10, 0.1)
