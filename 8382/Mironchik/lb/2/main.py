import pandas
from tensorflow.keras.layers import Dense
from tensorflow.keras.models import Sequential
from sklearn.preprocessing import LabelEncoder
import matplotlib.pyplot as plot


def plot_single_history(history, color="blue"):
    keys = ["loss", "accuracy", "val_loss", "val_accuracy"]
    titles = ["Loss", "Accuracy", "Val loss", "Val accuracy"]
    xlabels = ["epoch", "epoch", "epoch", "epoch"]
    ylabels = ["loss", "accuracy", "loss", "accuracy"]
    ylims = [3, 1.1, 3, 1.1]
    for i in range(len(keys)):
        plot.subplot(2, 2, i + 1)
        plot.title(titles[i])
        plot.xlabel(xlabels[i])
        plot.ylabel(ylabels[i])
        plot.gca().set_ylim([0, ylims[i]])
        plot.grid()
        values = history[keys[i]]
        plot.plot(range(len(values)), values, color=color)


dataframe = pandas.read_csv("sonar.csv", header=None)
dataset = dataframe.values
X = dataset[:, 0:30].astype(float)
Y = dataset[:, 60]

encoder = LabelEncoder()
encoder.fit(Y)
encoded_Y = encoder.transform(Y)

model = Sequential()
model.add(Dense(60, input_dim=30, activation='relu'))
model.add(Dense(15, activation='relu'))
model.add(Dense(1, activation='sigmoid'))

model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
H = model.fit(X, encoded_Y, epochs=100, batch_size=10, validation_split=0.1)

plot_single_history(H.history)
plot.show()


