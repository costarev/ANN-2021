import numpy as np
from tensorflow.keras.datasets import boston_housing
from tensorflow.keras.layers import Dense
from tensorflow.keras.models import Sequential
import matplotlib.pyplot as plt


def build_model():
    model = Sequential()
    model.add(Dense(64, activation='relu', input_shape=(train_data.shape[1],)))
    model.add(Dense(64, activation='relu'))
    model.add(Dense(1))
    model.compile(optimizer='rmsprop', loss='mse', metrics=['mae'])
    return model


def paint_plots(epochs, mae, val_mae, label):
    plt.plot(epochs, mae, 'b', label='Training ' + label)
    plt.plot(epochs, val_mae, 'k', label='Validation ' + label)
    plt.title('Training and validation ' + label)
    plt.xlabel('Epochs')
    plt.ylabel(label)
    plt.legend()
    plt.show()


(train_data, train_targets), (test_data, test_targets) = boston_housing.load_data()
mean = train_data.mean(axis=0)
std = train_data.std(axis=0)

train_data -= mean
train_data /= std

test_data -= mean
test_data /= std

k = 8
num_val_samples = len(train_data) // k
num_epochs = 40
all_scores = []
all_mae = np.zeros(num_epochs)
all_val_mae = np.zeros(num_epochs)
epochs = range(1, num_epochs + 1)

for i in range(k):
    print('processing fold #', i)
    val_data = train_data[i * num_val_samples: (i + 1) * num_val_samples]
    val_targets = train_targets[i * num_val_samples: (i + 1) * num_val_samples]
    partial_train_data = np.concatenate(
        [train_data[:i * num_val_samples],
         train_data[(i + 1) * num_val_samples:]],
        axis=0)

    partial_train_targets = np.concatenate(
        [train_targets[:i * num_val_samples],
         train_targets[(i + 1) * num_val_samples:]],
        axis=0)

    model = build_model()
    history_dict = model.fit(partial_train_data, partial_train_targets, epochs=num_epochs, batch_size=1, verbose=0,
                             validation_data=(val_data, val_targets)).history

    val_mse, val_mae = model.evaluate(val_data, val_targets, verbose=0)
    all_scores.append(val_mae)

    all_mae += np.array(history_dict['mae'])
    all_val_mae += np.array(history_dict['val_mae'])

    paint_plots(epochs, history_dict['mae'], history_dict['val_mae'], 'mae')
    plt.clf()
    paint_plots(epochs, history_dict['loss'], history_dict['val_loss'], 'mse')


paint_plots(epochs, all_mae / k, all_val_mae / k, 'mean mae')

print(np.mean(all_scores))
