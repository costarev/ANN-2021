import numpy as np
from tensorflow.keras.layers import Dense
from tensorflow.keras.models import Sequential

import matplotlib.pyplot as plt
from tensorflow.keras.datasets import boston_housing

(train_data, train_targets), (test_data, test_targets) = boston_housing.load_data()

mean = train_data.mean(axis=0)
std = train_data.std(axis=0)

train_data -= mean
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
num_epochs = 40
all_scores = []

for i in range(k):
    print('processing fold #', i)
    val_data = train_data[i * num_val_samples: (i + 1) * num_val_samples]
    val_targets = train_targets[i * num_val_samples: (i + 1) * num_val_samples]

    partial_train_data = np.concatenate([train_data[:i * num_val_samples],
                                         train_data[(i + 1) * num_val_samples:]], axis=0)
    partial_train_target = np.concatenate([train_targets[: i * num_val_samples],
                                           train_targets[(i + 1) * num_val_samples:]], axis=0)

    model = build_model()
    history = model.fit(partial_train_data, partial_train_target, epochs=num_epochs, batch_size=1,
                        validation_data=(val_data, val_targets), verbose=2)
    res_mse, res_mae = model.evaluate(test_data, test_targets)
    print(res_mae)
    mae = history.history['mae']
    val_mae = history.history['val_mae']
    mse = history.history['loss']
    val_mse = history.history['val_loss']
    x = range(1, num_epochs + 1)
    all_scores.append(val_mae)
    plt.figure(i)
    plt.plot(x, mae, 'r', label='Training MAE')
    plt.plot(x, val_mae, 'g', label='Validation MAE')
    plt.xlabel('Epochs')
    plt.ylabel('Absolute error')
    plt.title('Absolute error')
    plt.legend()
    plt.grid()
    plt.figure(k + i + 1)
    plt.plot(x, mse, 'r', label='Training MSE')
    plt.plot(x, val_mse, 'g', label='Validation MSE')
    plt.xlabel('Epochs')
    plt.ylabel('MSE')
    plt.title('MSE')
    plt.legend()
    plt.grid()

avg_mae = [np.mean([x[i] for x in all_scores]) for i in range(num_epochs)]

plt.figure(k)
plt.plot(range(1, num_epochs + 1), avg_mae, 'g')
plt.xlabel('Epochs')
plt.ylabel("Mean absolute error")
plt.grid()
figs = [plt.figure(n) for n in plt.get_fignums()]
for i in range(len(figs)):
    figs[i].savefig("./f%d.png" % i, format='png')

print(np.mean(all_scores))
