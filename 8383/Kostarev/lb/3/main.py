import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.layers import Dense
from tensorflow.keras.models import Sequential
from tensorflow.keras.datasets import boston_housing


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
num_epochs = 35
all_scores = []
all_mae = np.zeros(num_epochs)
all_val_mae = np.zeros(num_epochs)
epochs = range(1, num_epochs + 1)

for i in range(k):
    print('processing fold #', i)
    val_data = train_data[i * num_val_samples: (i + 1) * num_val_samples]
    val_targets = train_targets[i * num_val_samples: (i + 1) * num_val_samples]
    partial_train_data = np.concatenate([train_data[:i * num_val_samples], train_data[(i + 1) * num_val_samples:]],
                                        axis=0)
    partial_train_targets = np.concatenate(
        [train_targets[:i * num_val_samples], train_targets[(i + 1) * num_val_samples:]], axis=0)
    model = build_model()
    history = model.fit(partial_train_data, partial_train_targets, epochs=num_epochs, batch_size=1, verbose=0,
                        validation_data=(val_data, val_targets)).history
    val_mse, val_mae = model.evaluate(val_data, val_targets, verbose=0)
    all_scores.append(val_mae)

    all_mae += np.array(history['mae'])
    all_val_mae += np.array(history['val_mae'])
    all_scores.append(val_mae)

    all_mae += np.array(history['mae'])
    all_val_mae += np.array(history['val_mae'])

    plt.plot(epochs, history['mae'], 'b', label='Training')
    plt.plot(epochs, history['val_mae'], 'r', label='Validation')
    plt.title('Training and validation mae')
    plt.xlabel('Epochs')
    plt.ylabel('mae')
    plt.legend()
    plt.show()
    plt.clf()

    plt.plot(epochs, history['loss'], 'b', label='Training')
    plt.plot(epochs, history['val_loss'], 'r', label='Validation')
    plt.title('Training and validation mse')
    plt.xlabel('Epochs')
    plt.ylabel('mae')
    plt.legend()
    plt.show()


plt.plot(epochs, all_mae / k, 'b', label='Training')
plt.plot(epochs, all_val_mae / k, 'r', label='Validation')
plt.title('Mean of mae')
plt.xlabel('Epochs')
plt.ylabel('Mean')
plt.legend()
plt.show()

print(np.mean(all_scores))
