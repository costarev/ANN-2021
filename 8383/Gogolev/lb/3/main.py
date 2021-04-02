# coding: utf-8
import pandas
import matplotlib.pyplot as plt
import numpy as np
from tensorflow.keras.layers import Dense
from tensorflow.keras.models import Sequential
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.datasets import boston_housing

def plot_results(history, filename=""):
    loss = history.history['loss']
    val_loss = history.history['val_loss']
    acc = history.history['accuracy']
    val_acc = history.history['val_accuracy']
    epochs = range(1, len(loss) + 1)
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(20,20))
    fig.suptitle('Loss and accuracy')
    ax1.plot(epochs, loss, color="green", label='Training loss')
    ax1.plot(epochs, val_loss, color="blue", label='Validation loss')
    ax1.legend()
    ax2.plot(epochs, acc, color="green", label='Training acc')
    ax2.plot(epochs, val_acc, color="blue", label='Validation acc')
    ax2.legend()
    plt.show()
    if (filename != ""):
        plt.savefig(filename)

def print_average(mae_arr, val_mae_arr):
    avg_mae = np.average(mae_arr, axis=0)
    avg_val_mae = np.average(val_mae_arr, axis=0)
    epochs = range(1, len(avg_mae) + 1)
    plt.plot(epochs, avg_mae, color='green', label='Training mae')
    plt.plot(epochs, avg_val_mae, color='blue', label='Validation mae')
    plt.title('Average mae')
    plt.legend()
    plt.show()

def build_model():
    model = Sequential()
    model.add(Dense(64, activation='relu', input_shape=(train_data.shape[1],)))
    model.add(Dense(64, activation='relu'))
    model.add(Dense(1))
    model.compile(optimizer='rmsprop', loss='mse', metrics=['mae'])
    return model

(train_data, train_targets), (test_data, test_targets) = boston_housing.load_data()
mean = train_data.mean(axis=0)
std = train_data.std(axis=0)
train_data -= mean
train_data /= std
test_data -= mean
test_data /= std

k = 4
num_val_samples = len(train_data) // k
num_epochs = 100
all_scores = []
mae_arr = []
val_mae_arr = []

for i in range(k):
    print('processing fold #', i)
    val_data = train_data[i * num_val_samples: (i + 1) * num_val_samples]
    val_targets = train_targets[i * num_val_samples: (i + 1) * num_val_samples]
    p_train_data = np.concatenate([train_data[:i * num_val_samples], train_data[(i + 1) * num_val_samples:]], axis=0)
    p_train_targets = np.concatenate([train_targets[:i * num_val_samples], train_targets[(i + 1) * num_val_samples:]], axis=0)
    
    model = build_model()
    history = model.fit(p_train_data, p_train_targets, epochs=num_epochs, batch_size=1, verbose=False, validation_data=(val_data, val_targets))
    
    val_mse, val_mae = model.evaluate(val_data, val_targets, verbose=0)
    all_scores.append(val_mae)
    mae_arr.append(history.history['mae'])
    val_mae_arr.append(history.history['val_mae'])

print(np.mean(all_scores))
print_average(mae_arr, val_mae_arr)
