import numpy as np
from tensorflow.keras.layers import Dense
from tensorflow.keras.models import Sequential
from tensorflow.keras.datasets import boston_housing
import matplotlib.pyplot as plt


def build_model():
    model = Sequential()
    model.add(Dense(64, activation='relu', input_shape=(train_data.shape[1],)))
    model.add(Dense(64, activation='relu'))
    model.add(Dense(1))
    model.compile(optimizer='rmsprop', loss='mse', metrics=['mae'])
    return model


def draw_plot(history):
    keys = ["loss", "mae", "val_loss", "val_mae"]
    y = ["loss", "mae", "loss", "mae"]

    for i in range(len(keys)):
        plt.subplot(2, 2, i + 1)
        plt.title(keys[i])
        plt.xlabel("Epoch")
        plt.ylabel(y[i])
        plt.grid()
        values = history[keys[i]]
        plt.plot(range(1, len(values) + 1), values)


(train_data, train_targets), (test_data, test_targets) = boston_housing.load_data()
mean = train_data.mean(axis=0)
train_data -= mean
std = train_data.std(axis=0)
train_data /= std
test_data -= mean
test_data /= std

k = 5
num_val_samples = len(train_data) // k
num_epochs = 55
all_scores = []

all_loss = []
all_loss_validation = []
all_mae = []
all_mae_validation = []

for i in range(k):
    print('processing fold #', i)
    val_data = train_data[i * num_val_samples: (i + 1) * num_val_samples]
    val_targets = train_targets[i * num_val_samples: (i + 1) * num_val_samples]
    partial_train_data = np.concatenate([train_data[:i * num_val_samples], train_data[(i + 1) * num_val_samples:]], axis=0)
    partial_train_targets = np.concatenate(
        [train_targets[:i * num_val_samples], train_targets[(i + 1) * num_val_samples:]], axis=0)
    model = build_model()
    h = model.fit(partial_train_data, partial_train_targets, epochs=num_epochs, batch_size=1, verbose=0, validation_data=(val_data, val_targets))
    val_mse, val_mae = model.evaluate(val_data, val_targets, verbose=0)
    all_scores.append(val_mae)
    plt.suptitle('Модель {}'.format(i))
    draw_plot(h.history)
    plt.show()
    all_loss.append(h.history["loss"])
    all_loss_validation.append(h.history["val_loss"])
    all_mae.append(h.history["mae"])
    all_mae_validation.append(h.history["val_mae"])


print(np.mean(all_scores))
plt.suptitle('Средний результат')
draw_plot({"loss": np.mean(all_loss, axis=0), "val_loss": np.mean(all_loss_validation, axis=0), "mae": np.mean(all_mae, axis=0), "val_mae": np.mean(all_mae_validation, axis=0)})
plt.show()
