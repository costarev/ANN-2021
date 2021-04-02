import numpy as np
from tensorflow.keras.layers import Dense
from tensorflow.keras.models import Sequential
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.datasets import boston_housing
import matplotlib.pyplot as plt


(train_data, train_targets), (test_data, test_targets) = boston_housing.load_data()

mean = train_data.mean(axis=0)
train_data -= mean
std = train_data.std(axis=0)
train_data /= std

test_data -= mean
test_data /= std


def build_model():
    model = Sequential()
    model.add(Dense(64, activation='relu', input_shape=(train_data.shape[1],)))
    model.add(Dense(64, activation='relu'))
    model.add(Dense(1))
    model.compile(optimizer='rmsprop', loss='mse', metrics=['mae'])
    return model


k = 6
num_val_samples = len(train_data) // k
num_epochs = 60

all_loss = []
all_mae = []
histories = []


for i in range(k):
    print('processing fold #', i)

    val_data = train_data[i * num_val_samples: (i + 1) * num_val_samples]
    val_targets = train_targets[i * num_val_samples: (i + 1) * num_val_samples]

    partial_train_data = np.concatenate([
        train_data[:i * num_val_samples], train_data[(i + 1) * num_val_samples:]], axis=0)
    partial_train_targets = np.concatenate(
        [train_targets[:i * num_val_samples], train_targets[(i + 1) * num_val_samples:]], axis=0)

    model = build_model()
    history = model.fit(partial_train_data, partial_train_targets, epochs=num_epochs, batch_size=1, verbose=0,
                        validation_data=(val_data, val_targets))

    all_loss.append(history.history['val_loss'])
    all_mae.append(history.history['val_mae'])

    histories.append(history)


print(np.mean(all_mae))


for history in histories:
    history_dict = history.history

    mse_values = history_dict['loss']
    val_mse_values = history_dict['val_loss']
    epochs = range(1, len(mse_values) + 1)
    plt.plot(epochs, mse_values, 'blue', label='Training loss', linewidth=2.5)
    plt.plot(epochs, val_mse_values, 'orange', label='Validation loss', linewidth=2.5)
    plt.title('Training and validation loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.show()

    plt.clf()
    mae_values = history_dict['mae']
    val_mae_values = history_dict['val_mae']
    plt.plot(epochs, mae_values, 'blue', label='Training MAE', linewidth=2.5)
    plt.plot(epochs, val_mae_values, 'red', label='Validation MAE', linewidth=2.5)
    plt.title('Training and validation MAE')
    plt.xlabel('Epochs')
    plt.ylabel('MAE')
    plt.legend()
    plt.show()


avg_mae = np.asarray(all_mae).mean(axis=0)
avg_loss = np.asarray(all_loss).mean(axis=0)

epochs = range(1, len(avg_mae) + 1)
plt.plot(epochs, avg_loss, 'orange', linewidth=2.5)
plt.title('Average validation loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.show()

plt.clf()
plt.plot(epochs, avg_mae, 'red', linewidth=2.5)
plt.title('Average validation MAE')
plt.xlabel('Epochs')
plt.ylabel('MAE')
plt.show()
