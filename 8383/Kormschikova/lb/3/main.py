import numpy as np
import matplotlib.pyplot as plt  #импорт модуля для графиков
from tensorflow.keras.layers import Dense
from tensorflow.keras.models import Sequential
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.datasets import boston_housing


def build_model():
    model = Sequential()
    model.add(Dense(64, activation='relu', input_shape=(train_data.shape[1],)))
    model.add(Dense(64, activation='relu'))
    model.add(Dense(1))
    model.compile(optimizer='rmsprop', loss='mse', metrics=['mae'])
    return model


def plot_mae(history):
    color = "bgrmcyk"
    for i in range(len(history)):
        history_dict = history[i].history
        mae_values = history_dict['mae']
        val_mae_values = history_dict['val_mae']
        epochs = range(1, len(mae_values) + 1)
        tmpstr = 'Training mae '+str(i+1)
        tmpstr_2 = 'Validation mae '+str(i+1)
        plt.plot(epochs, mae_values, color=color[i], linestyle="-", linewidth=1, label=tmpstr)
        plt.plot(epochs, val_mae_values, color=color[i], linestyle="-.", linewidth=1, label=tmpstr_2)
    plt.title('Training and validation mae')
    plt.xlabel('Epochs')
    plt.ylabel('Mean absolute error')
    plt.legend()
    plt.show()


def plot_avg_mae(history, epochs):
    avg_mae = np.zeros(epochs)
    avg_val_mae = np.zeros(epochs)
    avg_mse = np.zeros(epochs)
    avg_val_mse = np.zeros(epochs)
    for i in range(len(history)):
        history_dict = history[i].history
        avg_mae += history_dict['mae']
        avg_mse += history_dict['loss']
        avg_val_mae += history_dict['val_mae']
        avg_val_mse += history_dict['val_loss']
    avg_mae /= len(history)
    avg_val_mae /= len(history)
    avg_mse /= len(history)
    avg_val_mse /= len(history)
    epochs_ = range(1, epochs + 1)
    #msa
    plt.plot(epochs_, avg_mae, color='g', linestyle="-", linewidth=1, label='Avg training mae')
    plt.plot(epochs_, avg_val_mae, color='b', linestyle="-.", linewidth=1, label='Avg validation mae')
    plt.title('Average training and validation mae')
    plt.xlabel('Epochs')
    plt.ylabel('Mean absolute error')
    plt.legend()
    plt.show()
    #mse
    plt.clf()
    plt.plot(epochs_, avg_mse, color='g', linestyle="-", linewidth=1, label='Avg training mse')
    plt.plot(epochs_, avg_val_mse, color='b', linestyle="-.", linewidth=1, label='Avg validation mse')
    plt.title('Average training and validation mse')
    plt.xlabel('Epochs')
    plt.ylabel('Mean squared error')
    plt.legend()
    plt.show()

(train_data, train_targets), (test_data, test_targets) = boston_housing.load_data()
# print(train_data.shape)
# print(test_data.shape)
# print(test_targets)
mean = train_data.mean(axis=0)
train_data -= mean
std = train_data.std(axis=0)
train_data /= std
test_data -= mean
test_data /= std

k = 6
num_val_samples = len(train_data) // k
print(num_val_samples)
num_epochs = 40
all_scores = []
all_history = []
for i in range(k):
    print('processing fold #', i)
    #проверочные данные
    val_data = train_data[i * num_val_samples: (i + 1) * num_val_samples] #i часть данных
    val_targets = train_targets[i * num_val_samples: (i + 1) * num_val_samples]
    #обучающие данные
    partial_train_data = np.concatenate([train_data[:i * num_val_samples], train_data[(i + 1) * num_val_samples:]], axis=0)#данные без i части
    partial_train_targets = np.concatenate([train_targets[:i * num_val_samples], train_targets[(i + 1) * num_val_samples:]], axis=0)
    model = build_model()
    history = model.fit(partial_train_data, partial_train_targets, epochs=num_epochs, batch_size=1,   validation_data=(val_data, val_targets), verbose=0)
    val_mse, val_mae = model.evaluate(val_data, val_targets, verbose=0)
    all_scores.append(val_mae)
    all_history.append(history)

print(np.mean(all_scores))
plot_mae(all_history)
plot_avg_mae(all_history, num_epochs)
