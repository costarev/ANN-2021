import numpy as np
from tensorflow.keras.layers import Dense
from tensorflow.keras.models import Sequential
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.datasets import boston_housing
import matplotlib.pyplot as plt


def build_model():
    model = Sequential()
    model.add(Dense(64, activation='relu', input_shape=(train_data.shape[1],)))
    model.add(Dense(64, activation='relu'))
    model.add(Dense(1))
    model.compile(optimizer='rmsprop', loss='mse', metrics=['mae'])
    return model


def plot_mae(history):
    for i in range(len(history)):
        hist_mae = history[i].history["mae"]
        hist_mae_val = history[i].history["val_mae"]
        hist_loss = history[i].history["loss"]
        hist_loss_val = history[i].history["val_loss"]
        epochs_vals = range(1, len(hist_mae) + 1)
        color1 = np.random.uniform(0, 1, 3)
        color2 = np.random.uniform(0, 1, 3)
        plt.subplot(2, 1, 1)
        plt.plot(epochs_vals, hist_mae, color=color1, linestyle="dotted")
        plt.plot(epochs_vals, hist_mae_val, color=color2)
        plt.subplot(2, 1, 2)
        plt.plot(epochs_vals, hist_loss, color=color1, linestyle="dotted")
        plt.plot(epochs_vals, hist_loss_val, color=color2)
    plt.subplot(2, 1, 1)
    plt.title("MAE")
    plt.xlabel("Epochs")
    plt.ylabel("Mean absolute error")
    plt.legend(("Training", "Validation"))
    plt.subplot(2, 1, 2)
    plt.legend(("Training", "Validation"))
    plt.title("Loss")
    plt.ylabel("Loss")

    plt.tight_layout()
    plt.show()


def plot_avg_mae(history):
    avg_mae = np.zeros(num_epochs)
    avg_validation_mae = np.zeros(num_epochs)
    avg_loss = np.zeros(num_epochs)
    avg_validation_loss = np.zeros(num_epochs)
    for i in range(1, len(history)):
        avg_mae += history[i].history["mae"]
        avg_validation_mae += history[i].history["val_mae"]
        avg_loss += history[i].history["loss"]
        avg_validation_loss += history[i].history["val_loss"]
    avg_mae /= len(history)
    avg_validation_mae /= len(history)
    avg_loss /= len(history)
    avg_validation_loss /= len(history)
    epochs = range(1, num_epochs + 1)

    # min_mae_index = np.argmin(avg_validation_mae)

    plt.subplot(2, 1, 1)
    plt.plot(epochs, avg_mae, color="b", label="Average training mae", linestyle="dotted")
    plt.plot(epochs, avg_validation_mae, color="r", label="Average validation mae")
    # plt.plot(epochs[min_mae_index], avg_validation_mae[min_mae_index], "go")
    plt.title("Average training mae")
    plt.xlabel("Epochs")
    plt.ylabel("Mean absolute error")
    plt.legend()
    plt.subplot(2, 1, 2)
    plt.plot(epochs, avg_loss, color="b", label="Average training loss", linestyle="dotted")
    plt.plot(epochs, avg_validation_loss, color="r", label="Average validation loss")
    plt.title("Average training loss")
    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    plt.legend()

    plt.tight_layout()
    plt.show()

(train_data, train_targets), (test_data, test_targets) = boston_housing.load_data()


mean = train_data.mean(axis=0)
train_data -= mean
std = train_data.std(axis=0)
train_data /= std
test_data -= mean
test_data /= std

k = 12
num_val_samples = len(train_data) // k
num_epochs = 75
all_scores = []
all_history = []

for i in range(k):
    print('processing fold #', i)
    val_data = train_data[i * num_val_samples: (i + 1) * num_val_samples]
    val_targets = train_targets[i * num_val_samples: (i + 1) * num_val_samples]
    partial_train_data = np.concatenate([train_data[:i * num_val_samples], 
                                        train_data[(i + 1) * num_val_samples:]], axis=0)
    partial_train_targets = np.concatenate([train_targets[:i * num_val_samples], 
                                           train_targets[(i + 1) * num_val_samples:]], axis=0)
    model = build_model()
    history = model.fit(partial_train_data, partial_train_targets, epochs=num_epochs, batch_size=1, 
                        validation_data=(val_data, val_targets), verbose=0)
    val_mse, val_mae = model.evaluate(val_data, val_targets, verbose=0)
    all_scores.append(val_mae)
    all_history.append(history)

print(np.mean(all_scores))
plot_mae(all_history)
plot_avg_mae(all_history)