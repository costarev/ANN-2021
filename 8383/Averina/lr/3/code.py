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
    model.add(Dense(1))  # линейный слой
    model.compile(optimizer='rmsprop', loss='mse', metrics=['mae'])
    return model

(train_data, train_targets), (test_data, test_targets) = boston_housing.load_data()
print(train_data.shape)
print(test_data.shape)

print(test_targets)

# Нормализация данных
mean = train_data.mean(axis=0)  # asix - ось, по которой выполняется среднее значение.
train_data -= mean
std = train_data.std(axis=0)
train_data /= std

test_data -= mean
test_data /= std

k = 6
num_val_samples = len(train_data) // k
num_epochs = 80
all_mae = []
all_loss = []
all_val_mae = []
all_val_loss = []
all_history = []

for i in range(k):
    print('processing fold #', i)
    val_data = train_data[i * num_val_samples: (i + 1) * num_val_samples]
    val_targets = train_targets[i * num_val_samples: (i + 1) * num_val_samples]

    partial_train_data = np.concatenate(
        [train_data[:i * num_val_samples], train_data[(i + 1) * num_val_samples:]],axis=0)
    partial_train_targets = np.concatenate(
        [train_targets[:i * num_val_samples], train_targets[(i + 1) * num_val_samples:]], axis=0)

    model = build_model()
    history = model.fit(partial_train_data, partial_train_targets,
                        epochs=num_epochs, batch_size=1, validation_data=(val_data, val_targets), verbose=0)

    val_loss, val_mae = model.evaluate(val_data, val_targets, verbose=0)

    all_val_mae.append(history.history['val_mae'])
    all_mae.append(history.history['mae'])
    plt.plot(history.history['mae'], 'r')
    plt.plot(history.history['val_mae'], 'b')
    plt.plot(range(num_epochs), np.full(num_epochs, min(history.history['val_mae'])), 'orange')
    plt.plot(np.argmin(history.history['val_mae']), min(history.history['val_mae']), 'o')
    plt.title('Mean accuracy' + ', i = ' + str(i + 1))
    plt.ylabel('mae')
    plt.xlabel('Epochs')
    plt.legend(['Training', 'Validation'], loc='upper left')
    plt.show()

    all_val_loss.append(history.history['val_loss'])
    all_loss.append(history.history['loss'])

    plt.plot(history.history['loss'], 'g')
    plt.plot(history.history['val_loss'], 'y')
    plt.plot(range(num_epochs), np.full(num_epochs, min(history.history['val_loss'])), 'orange')

    plt.title('Model loss' + ', i = ' + str(i + 1))
    plt.ylabel('loss')
    plt.xlabel('Epochs')
    plt.legend(['Training', 'Validation'], loc='upper left')
    plt.show()

print(np.mean(all_mae))


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