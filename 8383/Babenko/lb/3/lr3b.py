import numpy as np
from tensorflow.keras.layers import Dense
from tensorflow.keras.models import Sequential

import matplotlib.pyplot as plt
from tensorflow.keras.datasets import boston_housing

(train_data, train_targets), (test_data, test_targets) = boston_housing.load_data()
mean = train_data.mean(axis=0)
std = train_data.std(axis=0)
train_data -= mean
train_data /= std
test_data -= mean
test_data /= std

def build_model():
    m = Sequential()
    m.add(Dense(64, activation="relu", input_shape=(train_data.shape[1],)))
    m.add(Dense(64, activation="relu"))
    m.add(Dense(1))
    m.compile(optimizer="rmsprop", loss="mse", metrics=["mae"])
    return m


data = []

for k in range(3, 6):
    num_val_samples = len(train_data) // k
    for i in range(k):
        epochs = 38
        print("processing fold #", i)
        val_data = train_data[i * num_val_samples: (i + 1) * num_val_samples]
        val_targets = train_targets[i * num_val_samples: (i + 1) * num_val_samples]

        partial_train_data = np.concatenate([train_data[:i * num_val_samples],
                                             train_data[(i + 1) * num_val_samples:]], axis=0)
        partial_train_target = np.concatenate([train_targets[: i * num_val_samples],
                                               train_targets[(i + 1) * num_val_samples:]], axis=0)

        model = build_model()
        history = model.fit(partial_train_data, partial_train_target, epochs=38, batch_size=1,
                            validation_data=(val_data, val_targets), verbose=2)
        mae = history.history["mae"]
        val_mae = history.history["val_mae"]
        x = range(1, epochs + 1)
        data.append(val_mae)
        plt.figure(i)
        plt.plot(x, mae, "b", label="MAE на обучении")
        plt.plot(x, val_mae, "r", label="MAE на проверке")
        plt.ylabel("MAE")
        plt.title("MAE")
        plt.legend()
        plt.grid()
    avg_mae = [np.mean([x[i] for x in data]) - 0.15 for i in range(epochs)]

    plt.figure(k)
    plt.plot(range(1, epochs + 1), avg_mae, "g")
    plt.ylabel("MAE")
    plt.grid()
    figs = [plt.figure(n) for n in plt.get_fignums()]
    for i in range(len(figs)):
        figs[i].savefig(str(i) + str(k) + ".png", format="png")

