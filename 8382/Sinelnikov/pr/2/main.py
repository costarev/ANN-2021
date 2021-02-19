import numpy as np

import matplotlib.pyplot as plt

import matplotlib.colors as mclr

from tensorflow.keras import layers

from tensorflow.keras.optimizers import Adam

from tensorflow.keras import models

def genData(size=500):
    data = np.random.rand(size, 2) * 2 - 1
    label = np.zeros([size, 1])
    for i, p in enumerate(data):
        if p[0] * p[1] >= 0:
            label[i] = 1.
        else:
            label[i] = 0.

    div = round(size * 0.8)
    train_data = data[:div, :]
    test_data = data[div:, :]
    train_label = label[:div, :]
    test_label = label[div:, :]
    return (train_data, train_label), (test_data, test_label)


def drawResults(data, label, prediction):
    p_label = np.array([round(x[0]) for x in prediction])
    plt.scatter(data[:, 0], data[:, 1], s=30, c=label[:, 0], cmap=mclr.ListedColormap(['red', 'blue']))
    plt.scatter(data[:, 0], data[:, 1], s=10, c=p_label, cmap=mclr.ListedColormap(['red', 'blue']))
    plt.grid()
    plt.savefig("result.png")
    plt.show()


(train_data, train_label), (test_data, test_label) = genData()

mean = np.mean(train_data,axis=0)
std = np.std(train_data,axis=0)

standart_train_data = (train_data - mean) / std
standart_test_data = (test_data - mean) / std

model = models.Sequential()
model.add(layers.Dense(500, input_shape=(train_data.shape[1], )))
model.add(layers.Activation('relu'))
model.add(layers.BatchNormalization())
model.add(layers.Dropout(0.5))

model.add(layers.Dense(500))
model.add(layers.Activation('relu'))
model.add(layers.BatchNormalization())
model.add(layers.Dropout(0.5))

model.add(layers.Dense(1,activation="sigmoid"))

batch_size = 8
epochs = 50
model.compile(loss="binary_crossentropy", optimizer=Adam(lr=0.001,decay=1e-3/epochs), metrics=["accuracy"])

H = model.fit(standart_train_data, train_label, batch_size=batch_size, epochs=epochs, validation_split=0.1)


#Получение ошибки и точности в процессе обучения

loss = H.history['loss']
val_loss = H.history['val_loss']
acc = H.history['accuracy']
val_acc = H.history['val_accuracy']
epochs = range(1, len(loss) + 1)

#Построение графика ошибки

plt.plot(epochs, loss, 'bo', label='Training loss')
plt.plot(epochs, val_loss, 'b', label='Validation loss')
plt.title('Training and validation loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.savefig("loss.png")
plt.show()

#Построение графика точности

plt.clf()
plt.plot(epochs, acc, 'bo', label='Training acc')
plt.plot(epochs, val_acc, 'b', label='Validation acc')
plt.title('Training and validation accuracy')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend()
plt.savefig("acc.png")
plt.show()

#Получение и вывод результатов на тестовом наборе

results = model.evaluate(standart_test_data, test_label)
print("test results: r",results)

#Вывод результатов бинарной классификации
all_data = np.vstack((train_data, test_data))
all_label = np.vstack((train_label, test_label))
pred = model.predict(all_data)
drawResults(all_data, all_label, pred)