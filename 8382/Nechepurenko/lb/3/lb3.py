import numpy as np
from matplotlib import pyplot as plt
from tensorflow.keras.datasets import boston_housing
from tensorflow.keras.layers import Dense
from tensorflow.keras.models import Sequential


def next_prefix():
    while True:
        yield "b"


def next_postfix():
    i = 0
    while True:
        i += 1
        yield i


prefix_generator = next_prefix()
postfix_generator = next_postfix()


def save_fig():
    plt.savefig(f"images/{next(prefix_generator)}{next(postfix_generator)}.png")


def plot(epochs, train, validation, metrics):
    plt.plot(epochs, train, 'b', label=f'Training {metrics}')
    plt.plot(epochs, validation, 'r', label=f'Validation {metrics}')
    plt.title(f'Training and validation {metrics}')
    plt.xlabel('Epochs')
    plt.ylabel(metrics.upper())
    plt.grid(True)
    plt.legend()


def plot_history(history):
    loss = history['loss']
    val_loss = history['val_loss']
    acc = history['mae']
    val_acc = history['val_mae']
    epochs = range(1, len(loss) + 1)

    plt.figure()
    plot(epochs, loss, val_loss, "loss")
    save_fig()
    plt.figure()
    plot(epochs, acc, val_acc, "mae")
    save_fig()


def plot_mean(epochs, mean_list, name):
    plt.plot(epochs, mean_list, 'r', label=f'{name}')
    plt.title(f'Mean validation {name}')
    plt.xlabel('Epochs')
    plt.ylabel(name.upper())
    plt.grid(True)
    plt.legend()


def plot_means(loss_list, metrics_list):
    mean_loss_list = np.mean(loss_list, axis=0)
    mean_metrics_list = np.mean(metrics_list, axis=0)
    epochs = range(1, len(mean_loss_list) + 1)
    plt.figure()
    plot_mean(epochs, mean_loss_list, "mse")
    save_fig()
    plt.figure()
    plot_mean(epochs, mean_metrics_list, "mae")
    save_fig()


def read_data():
    (train_data, train_targets), (test_data, test_targets) = boston_housing.load_data()
    mean = train_data.mean(axis=0)
    train_data -= mean
    std = train_data.std(axis=0)
    train_data /= std
    test_data -= mean
    test_data /= std
    return (train_data, train_targets), (test_data, test_targets)


def build_model():
    model = Sequential()
    model.add(Dense(64, activation='relu', input_shape=(train_data.shape[1],)))
    model.add(Dense(64, activation='relu'))
    model.add(Dense(1))
    model.compile(optimizer='rmsprop', loss='mse', metrics=['mae'])
    return model


def cross_validation(k=4, epochs=100):
    num_val_samples = len(train_data) // k
    all_scores = []
    all_val_mae = []
    all_val_mse = []
    for i in range(k):
        print('processing fold #', i)
        val_data = train_data[i * num_val_samples: (i + 1) * num_val_samples]
        val_targets = train_targets[i * num_val_samples: (i + 1) * num_val_samples]
        partial_train_data = np.concatenate([train_data[:i * num_val_samples], train_data[(i + 1) * num_val_samples:]],
                                            axis=0)
        partial_train_targets = np.concatenate(
            [train_targets[:i * num_val_samples], train_targets[(i + 1) * num_val_samples:]], axis=0)
        model = build_model()
        history = model.fit(partial_train_data, partial_train_targets, epochs=epochs, batch_size=1,
                            validation_data=(val_data, val_targets), verbose=0)
        val_mse, val_mae = model.evaluate(val_data, val_targets, verbose=0)
        all_scores.append(val_mae)
        all_val_mae.append(history.history["val_mae"])
        all_val_mse.append(history.history["val_loss"])
        plot_history(history.history)
    print(np.mean(all_scores))
    plot_means(all_val_mae, all_val_mse)


(train_data, train_targets), (test_data, test_targets) = read_data()
cross_validation(5, 30)
