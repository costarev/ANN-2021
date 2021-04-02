import matplotlib.pyplot as plt
import numpy as np
from tensorflow.keras.datasets import boston_housing
from tensorflow.keras.layers import Dense
from tensorflow.keras.models import Sequential


def build_model():
    model = Sequential()
    model.add(Dense(64, activation='relu', input_shape=(train_data.shape[1],)))
    model.add(Dense(64, activation='relu'))
    # сеть способна предсказывать значения из любого диапазона
    model.add(Dense(1))
    # mean squared error, mean absolute error
    model.compile(optimizer='rmsprop', loss='mse', metrics=['mae'])
    return model


def graph(H, k):
    loss = H.history['loss']
    val_loss = H.history['val_loss']
    mae = H.history['mae']
    val_mae = H.history['val_mae']
    epochs = range(1, len(loss) + 1)

    plt.plot(epochs, loss, 'bo', label='Training loss')
    plt.plot(epochs, val_loss, 'b', label='Validation loss')
    plt.title('Training and validation loss, k = ' + str(k))
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.show()

    plt.clf()
    plt.plot(epochs, mae, 'bo', label='Training mae')
    plt.plot(epochs, val_mae, 'b', label='Validation mae')
    plt.title('Training and validation mae, k = ' + str(k))
    plt.xlabel('Epochs')
    plt.ylabel('Mae')
    plt.legend()
    plt.show()


def avg_graph(losses, val_losses, maes, val_maes):
    avg_losses = avg_list(losses)
    avg_val_losses = avg_list(val_losses)
    avg_maes = avg_list(maes)
    avg_val_maes = avg_list(val_maes)
    epochs = range(1, len(avg_losses) + 1)

    plt.plot(epochs, avg_losses, 'bo', label='Average loss')
    plt.plot(epochs, avg_val_losses, 'b', label='Average validation loss')
    plt.title('Average training and validation loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.show()

    plt.clf()
    plt.plot(epochs, avg_maes, 'bo', label='Average training mae')
    plt.plot(epochs, avg_val_maes, 'b', label='Average validation mae')
    plt.title('Average training and validation mae')
    plt.xlabel('Epochs')
    plt.ylabel('Mae')
    plt.legend()
    plt.show()


def avg_list(my_list):
    l = []

    for i in range(len(my_list[0])):
        summ = 0
        for j in range(len(my_list)):
            summ += my_list[j][i]
        l.append(summ / len(my_list))
    return l


(train_data, train_targets), (test_data, test_targets) = boston_housing.load_data()

print(train_data.shape)
print(test_data.shape)
print(test_targets)

# нормализация: из каждого значения вычитается среднее по этому признаку, и разность
# делится на стандартное отклонение, в результате признак центрируется по нулевому
# значению и имеет стандартное отклонение, равное единице
mean = train_data.mean(axis=0)
train_data -= mean
std = train_data.std(axis=0)
train_data /= std

test_data -= mean
test_data /= std

k = 4
num_val_samples = len(train_data) // k
num_epochs = 200
all_scores = []
losses = []
val_losses = []
maes = []
val_maes = []
models = []

for i in range(k):
    print('processing fold #', i)
    val_data = train_data[i * num_val_samples: (i + 1) * num_val_samples]
    val_targets = train_targets[i * num_val_samples: (i + 1) * num_val_samples]
    partial_train_data = np.concatenate([train_data[:i * num_val_samples], train_data[(i + 1) * num_val_samples:]],
                                        axis=0)
    partial_train_targets = np.concatenate(
        [train_targets[:i * num_val_samples], train_targets[(i + 1) * num_val_samples:]], axis=0)
    model = build_model()
    H = model.fit(partial_train_data, partial_train_targets, epochs=num_epochs, batch_size=1, verbose=0,
                  validation_data=(val_data, val_targets))
    val_mse, val_mae = model.evaluate(val_data, val_targets, verbose=0)
    all_scores.append(val_mae)
    losses.append(H.history['loss'])
    val_losses.append(H.history['val_loss'])
    maes.append(H.history['mae'])
    val_maes.append(H.history['val_mae'])
    graph(H, i)

print("Avg: " + str(np.mean(all_scores)))
print("All scores: " + str(all_scores))
print("Losses: " + str(losses))
avg_graph(losses, val_losses, maes, val_maes)
