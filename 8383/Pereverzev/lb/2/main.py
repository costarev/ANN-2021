import pandas
import matplotlib.pyplot as plt
from tensorflow.keras.layers import Dense
from tensorflow.keras.models import Sequential
from sklearn.preprocessing import LabelEncoder


def drawHistory(history):
    loss = history['loss']
    val_loss = history['val_loss']
    acc = history['accuracy']
    val_acc = history['val_accuracy']
    epochs = range(1, len(loss) + 1)
    fig, (ax1, ax2) = plt.subplots(1, 2)
    fig.suptitle('Loss and accuracy')

    ax1.plot(epochs, loss, color="b", label='Training loss')
    ax1.plot(epochs, val_loss, color="m", label='Validation loss')
    ax1.legend()

    ax2.plot(epochs, acc, color="b", label='Training acc')
    ax2.plot(epochs, val_acc, color="m", label='Validation acc')
    ax2.legend()

    plt.show()


dataframe = pandas.read_csv("sonar.csv", header=None)
dataset = dataframe.values
# X = dataset[:, 0:60].astype(float)
Y = dataset[:, 60]

encoder = LabelEncoder()
encoder.fit(Y)
encoded_Y = encoder.transform(Y)


def run(input_dim=60, layers=[60]):
    model = Sequential()
    model.add(Dense(layers[0], input_dim=input_dim, activation='relu'))
    for neurons in layers[1:]:
        model.add(Dense(neurons, activation='relu'))
    model.add(Dense(1, activation='sigmoid'))
    model.compile(optimizer='adam', loss='binary_crossentropy',
                            metrics=['accuracy'])

    history = model.fit(dataset[:, 0:input_dim].astype(float), encoded_Y, epochs=100,
                        batch_size=10, validation_split=0.1, verbose=0)
    drawHistory(history.history)


run()
run(50)
run(60, [30])
run(60, [60, 15])
run(40, [60, 40])
