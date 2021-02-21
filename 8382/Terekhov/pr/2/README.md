# Практика №2, вариант 3

## Задание

Необходимо дополнить следующий фрагмент кода моделью ИНС, которая способна провести бинарную классификацию по сгенерированным данным:

```python
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mclr
from tensorflow.keras import layers
from tensorflow.keras import models


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


def drawResults(data, label, prediction):
    p_label = np.array([round(x[0]) for x in prediction])
    plt.scatter(data[:, 0], data[:, 1], s=30, c=label[:, 0], cmap=mclr.ListedColormap(['red', 'blue']))
    plt.scatter(data[:, 0], data[:, 1], s=10, c=p_label, cmap=mclr.ListedColormap(['red', 'blue']))
    plt.grid()
    plt.show()


(train_data, train_label), (test_data, test_label) = genData()

#В данном месте необходимо создать модель и обучить ее

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
plt.show()

#Построение графика точности

plt.clf()
plt.plot(epochs, acc, 'bo', label='Training acc')
plt.plot(epochs, val_acc, 'b', label='Validation acc')
plt.title('Training and validation accuracy')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend()
plt.show()

#Получение и вывод результатов на тестовом наборе

results = model.evaluate(test_data, test_label)
print(results)

#Вывод результатов бинарной классификации

all_data = np.vstack((train_data, test_data))
all_label = np.vstack((train_label, test_label))
pred = model.predict(all_data)
drawResults(all_data, all_label, pred)
```

## Выполнение работы

В процессе работы было принято решение использовать следующую модель:
- два промежуточных слоя по 32 нейрона с функцией активации relu
- выходной слой с одним нейроном и функцией sigmoid

Параметры обучения:
- оптимизатор: rmsprop
- функция потерь: binary_crossentropy
- метрики: accuracy
- Число эпох: 128
- размер батча: 8

### Создание и обучение модели

```python
model = models.Sequential()
model.add(layers.Dense(32, activation='relu'))
model.add(layers.Dense(32, activation='relu'))
model.add(layers.Dense(1, activation='sigmoid'))
model.compile(optimizer='rmsprop', loss='binary_crossentropy', metrics=['accuracy'])
H = model.fit(train_data, train_label, epochs=128, batch_size=8, validation_data=(test_data, test_label), verbose=False)
```

## Графики

| ![](https://i.ibb.co/JvmzKTX/2021-02-21-17-52-28.png) |
|:---:|
| График потерь |

| ![](https://i.ibb.co/R7TmpZ0/2021-02-21-17-52-41.png) |
|:---:|
| График точности |

| ![](https://i.ibb.co/V3R0RLL/2021-02-21-17-52-51.png) |
|:---:|
|График результатов классификации|


## Выводы

Судя по графикам сеть успешно обучается примерно за 50 эпох, достигая почти точности в 100% и допуская умеренное количество ошибок. За 128 эпох в среднем на тестовых данных достигается точность в 99%, и ошибки не превышают значения в 0.05.