import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from tensorflow.keras.datasets import boston_housing
from tensorflow.keras.layers import Dense
from tensorflow.keras.models import Sequential


def build_model():
    model = Sequential()
    model.add(Dense(64, activation='relu', input_shape=(train_data.shape[1],)))
    model.add(Dense(64, activation='relu'))
    model.add(Dense(1))
    model.compile(optimizer='rmsprop', loss='mse', metrics=['mae'])
    return model


def crossval_cycle(k, num_epochs):
    num_val_samples = len(train_data) // k
    all_scores = []
    for i in range(k):
        print(f'processing fold #{i + 1}')
        val_data = train_data[i * num_val_samples: (i + 1) * num_val_samples]
        val_targets = train_targets[i * num_val_samples: (i + 1) * num_val_samples]
        partial_train_data = np.concatenate([train_data[:i * num_val_samples], train_data[(i + 1) * num_val_samples:]],
                                            axis=0)
        partial_train_targets = np.concatenate(
            [train_targets[:i * num_val_samples], train_targets[(i + 1) * num_val_samples:]], axis=0)
        model = build_model()
        fitted = model.fit(partial_train_data, partial_train_targets, epochs=num_epochs, batch_size=1,
                           validation_data=(val_data, val_targets), verbose=0)
        mae = pd.DataFrame(fitted.history)[['mae', 'val_mae']]
        all_scores.append(mae['val_mae'])
        plot(mae, "Absolute error", f"Fold #{i + 1} of {k} (num_epochs = {num_epochs})")
    mean_mae = [np.mean([x[j] for x in all_scores]) for j in range(num_epochs)]
    plot(pd.DataFrame(mean_mae, columns=['MAE']), "Mean absolute error",
         f"Mean absolute error(K = {k}, num_epochs = {num_epochs})")


def plot(data: pd.DataFrame, label: str, title: str):
    axis = sns.lineplot(data=data, dashes=False)
    axis.set(ylabel=label, xlabel='epochs', title=title.split('.')[0])
    axis.grid(True, linestyle="--")
    # plt.show()
    plt.savefig(f"img/{title.replace(' ', '_').replace('#', '').replace('.', '-')}_{label.replace(' ', '_')}")
    plt.clf()


if __name__ == '__main__':
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
    for i in [3, 4, 5]:
        crossval_cycle(i, 100)
    for i in [3, 4, 5]:
        crossval_cycle(i, 40)
