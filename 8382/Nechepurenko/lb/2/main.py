import pandas
from tensorflow.keras.layers import Dense
from tensorflow.keras.models import Sequential
from tensorflow.keras.utils import to_categorical
from sklearn.preprocessing import LabelEncoder
import matplotlib.pyplot as plt


def read_data():
    dataframe = pandas.read_csv("sample_data/sonar.csv", header=None)
    dataset = dataframe.values
    x = dataset[:, 0:60].astype(float)
    y = dataset[:, 60]
    return x, y


def encode_y(raw_y):
    encoder = LabelEncoder()
    encoder.fit(raw_y)
    encoded_y = encoder.transform(raw_y)
    y_mapping = dict(zip(encoder.classes_, encoder.transform(encoder.classes_)))
    return encoded_y, y_mapping


def plot(epochs, train, validation, metrics):
    plt.plot(epochs, train, 'b', label=f'Training {metrics}')
    plt.plot(epochs, validation, 'r', label=f'Validation {metrics}')
    plt.title(f'Training and validation {metrics}')
    plt.xlabel('Epochs')
    plt.ylabel(metrics.capitalize())
    plt.grid(True)
    plt.legend()


def plot_history(history):
    loss = history['loss']
    val_loss = history['val_loss']
    acc = history['accuracy']
    val_acc = history['val_accuracy']
    epochs = range(1, len(loss) + 1)

    plt.figure()
    plt.subplot(211)
    plot(epochs, loss, val_loss, "loss")
    plt.subplot(212)
    plot(epochs, acc, val_acc, "accuracy")
    plt.show()


def build_model():
    model = Sequential()
    model.add(Dense(60, input_dim=60, kernel_initializer='normal', activation='relu'))
    #model.add(Dense(30, input_dim=30, kernel_initializer='normal', activation='relu'))
    #model.add(Dense(30, kernel_initializer='normal', activation='relu'))
    #model.add(Dense(15, kernel_initializer='normal', activation='relu'))
    model.add(Dense(1, kernel_initializer='normal', activation='sigmoid'))

    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    return model


X, Y = read_data()
#X = X[:, :30]
Y, mapping = encode_y(Y)
model_ = build_model()
h = model_.fit(X, Y, epochs=100, batch_size=10, validation_split=0.1)
plot_history(h.history)
