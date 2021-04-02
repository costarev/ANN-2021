
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mclr
from tensorflow.keras.layers import Dense
from tensorflow.keras.models import Sequential


def genData(size=500):
    data = np.random.rand(size, 2)*2 - 1
    label = np.zeros([size, 1])
    for i, p in enumerate(data):
        if (p[0]+0.2)**2 + (0.6*p[1])**2 >= 0.25:
            label[i] = 0.
        else:
            label[i] = 1.
    div = round(size*0.8)
    train_data = data[:div, :]
    test_data = data[div:, :]
    train_label = label[:div, :]
    test_label = label[div:, :]
    return (train_data, train_label), (test_data, test_label)
    # Функцию выбрать в зависимости от варианта


def drawResults(data, label, prediction):
    p_label = np.array([round(x[0]) for x in prediction])
    plt.scatter(data[:, 0], data[:, 1], s=30, c=label[:, 0],
                cmap=mclr.ListedColormap(['red', 'blue']))
    plt.scatter(data[:, 0], data[:, 1], s=10, c=p_label,
                cmap=mclr.ListedColormap(['red', 'blue']))
    plt.grid()
    plt.show()


(train_data, train_label), (test_data, test_label) = genData()


# В данном месте необходимо создать модель и обучить ее
model = Sequential()
model.add(Dense(40, activation='relu', input_shape=(2,)))
model.add(Dense(24, activation='relu'))
model.add(Dense(1, activation='sigmoid'))

model.compile(optimizer='rmsprop', loss='binary_crossentropy',
              metrics=['accuracy'])
H = model.fit(train_data, train_label, epochs=50, batch_size=10,
              validation_data=(test_data, test_label), verbose=1)

# Получение ошибки и точности в процессе обучения

loss = H.history['loss']
val_loss = H.history['val_loss']
acc = H.history['accuracy']
val_acc = H.history['val_accuracy']
epochs = range(1, len(loss) + 1)

# Вывод результатов
# терминал
print(model.evaluate(test_data, test_label))

# Построение графиков
fig, axs = plt.subplots(2, 2)
fig.tight_layout(pad=3)

axs[0, 0].plot(epochs, val_loss, 'm', label='Validation loss')
axs[0, 0].set_title('Validation loss')
axs.flat[0].set(xlabel='', ylabel='Loss')


axs[0, 1].plot(epochs, loss, 'b', label='Training loss')
axs[0, 1].set_title('Training loss')
axs.flat[1].set(xlabel='', ylabel='')

axs[1, 0].plot(epochs, val_acc, 'm', label='Validation accuracy')
axs[1, 0].set_title('Validation accuracy')
axs.flat[2].set(xlabel='Epochs', ylabel='Accuracy')

axs[1, 1].plot(epochs, acc, 'b', label='Training accuracy')
axs[1, 1].set_title('Training accuracy')
axs.flat[3].set(xlabel='Epochs', ylabel='')

plt.show()
plt.clf()

# Вывод результатов
# график
resFullData = np.vstack((train_data, test_data))
resFullLabel = np.vstack((train_label, test_label))
drawResults(
    resFullData,
    resFullLabel,
    model.predict(resFullData)
)
