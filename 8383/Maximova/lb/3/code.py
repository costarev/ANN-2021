import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.layers import Dense               # полносвязанный слой
from tensorflow.keras.models import Sequential          # сеть прямого распространения
from tensorflow.keras.datasets import boston_housing    # данные

def build_model():
    model = Sequential()
    model.add(Dense(64, activation='relu', input_shape=(train_data.shape[1],)))
    model.add(Dense(64, activation='relu'))
    model.add(Dense(1))
    model.compile(optimizer='rmsprop', loss='mse', metrics=['mae'])
    return model

#Построение графика среднеквадратичной ошибки во время обучения
def plot_mse(mse_, val_mse_):                                          #график среднеквадратичной ошибки
    plt.clf()
    epochs = range(1, len(mse) + 1)
    plt.plot(epochs, mse_, label='Training MSE', linestyle='--', linewidth=2, color="darkmagenta")
    plt.plot(epochs, val_mse_, 'b', label='Validation MSE', color="lawngreen")
    plt.title('Training and validation MSE')                           #оглавление на рисунке
    plt.xlabel('Epochs')
    plt.ylabel('MSE')
    plt.legend()
    plt.show()

#Построение графика средней абсолютной ошибки во время обучения
def plot_mae(mae_, val_mae_):                                           #график  средней абсолютной ошибки
    plt.clf()
    epochs = range(1, len(mae) + 1)
    plt.plot(epochs, mae_, label='Training MAE', linestyle='--', linewidth=2, color="darkmagenta")
    plt.plot(epochs, val_mae_, 'b', label='Validation MAE', color="lawngreen")
    plt.title('Training and validation MAE')
    plt.xlabel('Epochs')
    plt.ylabel('MAE')
    plt.legend()
    plt.show()

def avg_val_mae(all_hist_, num_epochs_):
    avg_val_mae = np.zeros(num_epochs_)

    for i in range(len(all_hist_)):
        avg_val_mae += all_hist_[i].history['val_mae']

    avg_val_mae /= len(all_hist_)

    plot_avg_mae(avg_val_mae)

def plot_avg_mae(avg_val_mae_):               #график усредненной абсолютной ошибки
    plt.clf()
    epochs = range(1, len(mae) + 1)
    plt.plot(epochs, avg_val_mae_, 'b', label='Validation AVG MAE', color="indigo")
    plt.title('Validation AVG MAE')
    plt.xlabel('Epochs')
    plt.ylabel('VAL AVG MAE')
    plt.legend()
    plt.show()

#1-загрузка данных
(train_data, train_targets), (test_data, test_targets) = boston_housing.load_data()

mean = train_data.mean(axis=0)                          # среднее значение
std = train_data.std(axis=0)                            # стандартное отклонение

train_data -= mean
train_data /= std

test_data -= mean
test_data /= std

# перекрестная проверка по К блокам
k = 4
num_val_samples = len(train_data) // k
num_epochs = 50
all_scores = []   # массив оценок
all_hist = []

for i in range(k):
    print('processing fold #', i)
    val_data = train_data[i * num_val_samples: (i + 1) * num_val_samples]                   # 0-24, ..., 75-99
    val_targets = train_targets[i * num_val_samples: (i + 1) * num_val_samples]             # 0-24, ..., 75-99

    partial_train_data = np.concatenate([train_data[:i * num_val_samples],
                                         train_data[(i + 1) * num_val_samples:]], axis=0)   # пропуск i-ого блока
    partial_train_targets = np.concatenate([train_targets[:i * num_val_samples],
                                            train_targets[(i + 1) * num_val_samples:]], axis=0)
    # обучение
    model = build_model()
    hist = model.fit(partial_train_data, partial_train_targets, epochs=num_epochs, batch_size=1,
                        validation_data=(val_data, val_targets))

    mse = hist.history['loss']            # среднеквадратичная ошибка
    val_mse = hist.history['val_loss']
    mae = hist.history['mae']             # средняя абсолютная ошибка
    val_mae = hist.history['val_mae']

    all_scores.append(val_mae)
    all_hist.append(hist)
    plot_mse(mse, val_mse)
    plot_mae(mae, val_mae)

print(np.mean(all_scores))

avg_val_mae(all_hist, num_epochs)