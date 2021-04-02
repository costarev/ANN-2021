import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '1'
import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.datasets import boston_housing
import matplotlib.pyplot as plt
from matplotlib import gridspec


def print_plots(training_data):
    loss = training_data.history['loss']
    val_loss = training_data.history['val_loss']
    mae = training_data.history['mae']
    val_mae = training_data.history['val_mae']
    epochs = range(1, len(loss) + 1)

    fig = plt.figure(figsize=(12, 6))
    gs = gridspec.GridSpec(1, 2, width_ratios=[3, 3])
    plt.subplot(gs[0])
    plt.plot(epochs, loss, 'm-', label='Training')
    plt.plot(epochs, val_loss, 'b-', label='Validation')
    plt.title('Training and validation loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()

    plt.subplot(gs[1])
    plt.plot(epochs, mae, 'm-', label='Training')
    plt.plot(epochs, val_mae, 'b-', label='Validation')
    plt.title('Training and validation absolute error')
    plt.xlabel('Epochs')
    plt.ylabel('Mae')
    plt.legend()

    plt.tight_layout()
    plt.show()


def print_avg_plot(all_mae_history, num_epochs):
    average_mae = [np.mean([x[i] for x in all_mae_history]) for i in range(num_epochs)]
    plt.plot(range(1, num_epochs + 1), average_mae, 'c-')
    plt.xlabel('Epochs')
    plt.ylabel('MAE')
    plt.grid()
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


def start(k, num_epochs):
    num_val_samples = len(train_data) // k
    all_mae_H = []
    for i in range(k):
        print('processing fold #', i)
        val_data = train_data[i * num_val_samples: (i + 1) * num_val_samples]
        val_targets = train_targets[i * num_val_samples: (i + 1) * num_val_samples]
        partial_train_data = np.concatenate([train_data[:i * num_val_samples], train_data[(i + 1) * num_val_samples:]],
                                            axis=0)
        partial_train_targets = np.concatenate(
            [train_targets[:i * num_val_samples], train_targets[(i + 1) * num_val_samples:]], axis=0)
        model = build_model()
        H = model.fit(partial_train_data, partial_train_targets, validation_data=(val_data, val_targets), epochs=num_epochs,
                      batch_size=1, verbose=2)
        val_mae_history = H.history['val_mae']
        all_mae_H.append(val_mae_history)
        print_plots(H)
    print_avg_plot(all_mae_H, num_epochs)
    print(np.mean(all_mae_H))

start(3,50)