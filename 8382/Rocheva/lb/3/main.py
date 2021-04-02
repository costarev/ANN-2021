import numpy as np
from tensorflow.keras.layers import Dense
from tensorflow.keras.models import Sequential
from tensorflow.keras.datasets import boston_housing
import matplotlib.pyplot as plt


def show_loss(loss, val_loss):
    epochs = range(1, len(loss[0]) + 1)

    fig, ax = plt.subplots(2, 2)
    ax[0][0].plot(epochs, loss[0], 'g', label='Training loss')
    ax[0][0].plot(epochs, val_loss[0], 'r', label='Validation loss')
    ax[0][0].set_title('k = 1')
    ax[0][0].legend()
    ax[0][0].set_xlabel('epochs')
    ax[0][0].set_ylabel('mae')

    ax[0][1].plot(epochs, loss[1], 'g', label='Training loss')
    ax[0][1].plot(epochs, val_loss[1], 'r', label='Validation loss')
    ax[0][1].set_title('k = 2')
    ax[0][1].legend()
    ax[0][1].set_xlabel('epochs')
    ax[0][1].set_ylabel('mae')

    ax[1][0].plot(epochs, loss[2], 'g', label='Training loss')
    ax[1][0].plot(epochs, val_loss[2], 'r', label='Validation loss')
    ax[1][0].set_title('k = 3')
    ax[1][0].legend()
    ax[1][0].set_xlabel('epochs')
    ax[1][0].set_ylabel('mae')

    ax[1][1].plot(epochs, loss[3], 'g', label='Training loss')
    ax[1][1].plot(epochs, val_loss[3], 'r', label='Validation loss')
    ax[1][1].set_title('k = 4')
    ax[1][1].legend()
    ax[1][1].set_xlabel('epochs')
    ax[1][1].set_ylabel('mae')

    fig.set_figwidth(10)
    fig.set_figheight(10)

    plt.show()


def show_plots(mae, val_mae, loss, val_loss, title=''):
    epochs = range(1, len(loss) + 1)

    plt.plot(epochs, mae, 'g', label='Training mae')
    plt.plot(epochs, val_mae, 'r', label='Validation mae')
    plt.title(title + ' mae')
    plt.legend()
    plt.xlabel('epochs')
    plt.ylabel('mae')

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
train_data -= mean
std = train_data.std(axis=0)
train_data /= std

k = 9
num_val_samples = len(train_data) // k
num_epochs = 35
all_scores = []

array_loss = []
array_val_loss = []

for i in range(k):
    print('processing fold #', i)
    val_data = train_data[i * num_val_samples: (i + 1) * num_val_samples]
    val_targets = train_targets[i * num_val_samples: (i + 1) * num_val_samples]
    partial_train_data = np.concatenate([train_data[:i * num_val_samples], train_data[(i + 1) * num_val_samples:]], axis=0)
    partial_train_targets = np.concatenate([train_targets[:i * num_val_samples], train_targets[(i + 1) * num_val_samples:]], axis=0)
    model = build_model()
    H = model.fit(partial_train_data, partial_train_targets, epochs=num_epochs, batch_size=1, verbose=0, validation_data=(val_data, val_targets))
    val_mse, val_mae = model.evaluate(val_data, val_targets, verbose=0)
    all_scores.append(val_mae)
    array_loss.append(H.history['loss'])
    array_val_loss.append(H.history['val_loss'])
    label = 'Testing epochs (k = ' + str(i+1) + ')'
    show_plots(H.history['mae'], H.history['val_mae'], H.history['loss'], H.history['val_loss'], label)

# show_loss(array_loss, array_val_loss)
print(np.mean(all_scores))
