import numpy as np
from tensorflow.keras.layers import Dense
from tensorflow.keras.models import Sequential
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.datasets import boston_housing
import matplotlib.pyplot as plot


def plot_single_history(history, color="blue"):
    keys = ["loss", "mae", "val_loss", "val_mae"]
    titles = ["Loss", "Mae", "Val loss", "Val mae"]
    xlabels = ["epoch", "epoch", "epoch", "epoch"]
    ylabels = ["loss", "mae", "loss", "mae"]
    #ylims = [3, 1.1, 3, 1.1]

    for i in range(len(keys)):
        plot.subplot(2, 2, i + 1)
        plot.title(titles[i])
        plot.xlabel(xlabels[i])
        plot.ylabel(ylabels[i])
        #plot.gca().set_ylim([0, ylims[i]])
        plot.grid()
        values = history[keys[i]]
        plot.plot(range(1, len(values) + 1), values, color=color)


def build_model():
    model = Sequential()
    model.add(Dense(64, activation='relu', input_shape=(train_data.shape[1],)))
    model.add(Dense(64, activation='relu'))
    model.add(Dense(1))
    model.compile(optimizer='rmsprop', loss='mse', metrics=['mae'])
    return model


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

k = 6
num_val_samples = len(train_data) // k
num_epochs = 70
all_mae = []
all_loss = []
all_val_mae = []
all_val_loss = []

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
              batch_size=1, verbose=0)
    plot.suptitle('Model {0} of {1}. Epochs: {2}'.format(i + 1, k, num_epochs))
    plot_single_history(H.history)
    plot.show()
    all_mae.append(H.history['mae'])
    all_loss.append(H.history['loss'])
    all_val_mae.append(H.history['val_mae'])
    all_val_loss.append(H.history['val_loss'])

plot.suptitle("Mean results. k: {0}, epochs: {1}".format(k, num_epochs))
plot_single_history({
    'mae': np.mean(all_mae, axis=0),
    'val_mae': np.mean(all_val_mae, axis=0),
    'loss': np.mean(all_loss, axis=0),
    'val_loss': np.mean(all_val_loss, axis=0),
})
plot.show()
print(np.mean(np.asarray(all_val_mae)[:, -1]))
