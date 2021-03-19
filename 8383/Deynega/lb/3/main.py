import pandas
from tensorflow.keras.layers import Dense
from tensorflow.keras.models import Sequential
from sklearn.preprocessing import LabelEncoder
import matplotlib.pyplot as plt
from tensorflow.keras.datasets import boston_housing
import numpy as np

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


k = 4

num_val_samples = len(train_data) // k
num_epochs = 35
all_scores = []
all_train_mae = np.zeros(num_epochs)
all_val_mae = np.zeros(num_epochs)

for i in range(k):
    print('processing fold #', i)
    val_data = train_data[i * num_val_samples: (i + 1) * num_val_samples]
    val_targets = train_targets[i * num_val_samples: (i + 1) * num_val_samples]
    partial_train_data = np.concatenate([train_data[:i * num_val_samples],
                                         train_data[(i + 1) * num_val_samples:]], axis=0)
    partial_train_targets = np.concatenate([train_targets[:i * num_val_samples],
                                            train_targets[(i + 1) * num_val_samples:]], axis=0)

    model = build_model()
    hist = model.fit(partial_train_data, partial_train_targets, epochs=num_epochs, batch_size=1, verbose=0,
                     validation_data=(val_data, val_targets))
    val_mse, val_mae = model.evaluate(val_data, val_targets, verbose=0)
    all_scores.append(val_mae)
    epochs = range(1, num_epochs + 1)
    all_train_mae += np.array(hist.history['mae'])
    all_val_mae += np.array(hist.history['val_mae'])
    plt.plot(epochs, hist.history['mae'], 'bo', label='Training MAE')
    plt.plot(epochs, hist.history['val_mae'], 'b', label='Validation MAE')
    plt.title('Training and validation mean absolute error')
    plt.xlabel('Epochs')
    plt.ylabel('MAE')
    plt.legend()
    plt.show()

    plt.clf()
    mse_values = hist.history['loss']
    val_mse_values = hist.history['val_loss']
    plt.plot(epochs, mse_values, 'bo', label='Training MSE')
    plt.plot(epochs, val_mse_values, 'b', label='Validation MSE')
    plt.title('Training and validation mean squared error')
    plt.xlabel('Epochs')
    plt.ylabel('MSE')
    plt.legend()
    plt.show()




print(np.mean(all_scores))

plt.clf()
plt.plot(epochs, all_train_mae/k, 'bo', label='Training mean MAE')
plt.plot(epochs, all_val_mae/k, 'b', label='Validation mean MAE')
plt.title('Training and validation mean MAE')
plt.xlabel('Epochs')
plt.ylabel('MAE')
plt.legend()
plt.show()
