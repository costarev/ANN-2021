import numpy as np
from tensorflow.keras.layers import Dense
from tensorflow.keras.models import Sequential
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.datasets import boston_housing
import matplotlib.pyplot as plt

(train_data, train_targets), (test_data, test_targets) = boston_housing.load_data()

print(train_data.shape)
print(test_data.shape)
print(test_targets)

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

k = 5
num_val_samples = len(train_data) // k
num_epochs = 35
all_scores = []
all_mae = []
arr = []

for i in range(k):

    print('processing fold #', i)
    val_data = train_data[i * num_val_samples: (i + 1) * num_val_samples]
    val_targets = train_targets[i * num_val_samples: (i + 1) * num_val_samples]
    partial_train_data = np.concatenate([train_data[:i * num_val_samples], train_data[(i + 1) * num_val_samples:]], axis=0)
    partial_train_targets = np.concatenate([train_targets[:i * num_val_samples], train_targets[(i + 1) * num_val_samples:]], axis=0)

    model = build_model()
    H = model.fit(partial_train_data, partial_train_targets, epochs=num_epochs, batch_size=1, validation_data=(val_data, val_targets), verbose=0)
    final_mse, final_mae = model.evaluate(val_data, val_targets, verbose=0) #средняя ошибка после последней эпохе
    all_scores.append(final_mae)

    # Получение ошибки и точности в процессе обучения

    loss = H.history['loss']
    val_loss = H.history['val_loss']
    mae = H.history['mean_absolute_error']
    val_mae = H.history['val_mean_absolute_error']
    all_mae.append(val_mae)

    epochs = range(1, len(loss) + 1)

    # Построение графика ошибки

    plt.plot(epochs, loss, 'm*', label='Training loss')
    plt.plot(epochs, val_loss, 'b', label='Validation loss')
    plt.title('Training and validation loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.show()

    # Построение графика mae

    plt.clf()
    plt.plot(epochs, mae, 'm*', label='Training mae')
    plt.plot(epochs, val_mae, 'b', label='Validation mae')
    plt.title('Training and validation mae')
    plt.xlabel('Epochs')
    plt.ylabel('MAE')
    plt.legend()
    plt.show()

# Построение итогового графика
for i in range(num_epochs):
    arr.append(np.mean([epochs[i] for epochs in all_mae]))
plt.plot(range(1, num_epochs + 1), arr, 'b')
plt.xlabel('Epochs')
plt.ylabel('MAE')
plt.show()

print(np.mean(all_scores))#вывод средней из итоговых средних ошибок


