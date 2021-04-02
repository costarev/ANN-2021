import numpy as np
from tensorflow.keras.layers import Dense
from tensorflow.keras.models import Sequential
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.datasets import boston_housing
import matplotlib.pyplot as plt


# Построение модели
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
test_data -= mean
test_data /= std


k = 7
num_val_samples = len(train_data) // k
num_epochs = 50
allValLoss = []
allValMae = []

for i in range(k):
    print('processing fold #', i)
    val_data = train_data[i * num_val_samples: (i + 1) * num_val_samples]
    val_targets = train_targets[i * num_val_samples: (i + 1) * num_val_samples]
    partial_train_data = np.concatenate([train_data[:i * num_val_samples], train_data[(i + 1) * num_val_samples:]],
                                        axis=0)
    partial_train_targets = np.concatenate(
        [train_targets[:i * num_val_samples], train_targets[(i + 1) * num_val_samples:]], axis=0)
    model = build_model()
    history = model.fit(partial_train_data, partial_train_targets, validation_data = (val_data, val_targets),
                        epochs=num_epochs, batch_size=1, verbose=0)

    loss = history.history['loss']
    val_loss = history.history['val_loss']
    mae = history.history['mae']
    val_mae = history.history['val_mae']
    epochs = range(1, len(mae) + 1)

    plt.plot(epochs, loss, 'bo', label='Training loss')
    plt.plot(epochs, val_loss, 'r', label='Validation loss')
    plt.title('Training and validation loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.show()

    plt.clf()
    plt.plot(epochs, mae, 'b', label='Training mae')
    plt.plot(epochs, val_mae, 'green', label='Validation mae')
    plt.title('Training and validation mae')
    plt.xlabel('Epochs')
    plt.ylabel('MAE')
    plt.legend()
    plt.show()

    val_loss = history.history["val_loss"]
    val_mae = history.history["val_mae"]
    allValMae.append(val_mae)
    allValLoss.append(val_loss)




allValLoss = np.asarray(allValLoss)
avg_loss = allValLoss.mean(axis = 0)
allValMae = np.asarray(allValMae)
avg_mae = allValMae.mean(axis = 0)
epochs = range(1, len(avg_mae) + 1)

plt.plot(epochs, avg_loss, 'green')
plt.title('Average validation loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.show()

plt.clf()
plt.plot(epochs, avg_mae, 'r')
plt.title('Average validation mae')
plt.xlabel('Epochs')
plt.ylabel('MAE')
plt.show()


print("MAE: ", np.mean(allValMae[:, -1]))
