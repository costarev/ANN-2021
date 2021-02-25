# Практическая работа №2

## Задание
Вариант 4.
Необходимо дополнить следующий фрагмент кода моделью ИНС, которая способна провести бинарную классификацию по сгенерированным данным:

```python
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mclr
from tensorflow.keras import layers
from tensorflow.keras import models

def genData(size=500):
    #Функцию выбрать в зависимости от варианта

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

Для варианта 4 предложена следующая функция генерации данных:
```python
def genData(size=500):
    size1 = size//2
    size2 = size - size1
    t1 = np.random.rand(size1)
    x1 = np.asarray([i * math.cos(i*2*math.pi) + (np.random.rand(1)-1)/2*i for i in t1])
    y1 = np.asarray([i * math.sin(i*2*math.pi) + (np.random.rand(1)-1)/2*i for i in t1])
    data1 = np.hstack((x1, y1))
    label1 = np.zeros([siz1, 1])
    div1 = round(size1*0.8)
    t2 = np.random.rand(size2)
    x2 = np.asarray([-i * math.cos(i*2*math.pi) + (np.random.rand(1)-1)/2*i for i in t2])
    y2 = np.asarray([-i * math.sin(i*2*math.pi) + (np.random.rand(1)-1)/2*i for i in t2])
    data2 = np.hstack((x2, y2))
    label2 = np.ones([size2, 1])
    div2 = round(size2*0.8)         
    div = div1 + div2
    order = np.random.permutation(div)
    train_data = np.vstack((data1[:div1], data2[:div2]))
    test_data = np.vstack((data1[div1:], data2[div2:]))
    train_label = np.vstack((label1[:div1], label2[:div2]))
    test_label = np.vstack((label1[div1:], label2[div2:]))
    return (train_data[order, :], train_label[order, :]), (test_data, test_label)
```

## Реализация

Создадим модель ИНС при помощи класса `Sequential` Keras API.
Добавим в нее два скрытых полносвязных слоя с функцией активации `relu`, по 32 нейрона на слой. Выходной слой будет состоять
из 1 полносвязного нейрона с сигмоидальной функцией активации.


В качестве оптимизатора выберем `adam`. Так как в генерируемом наборе данных оба класса имеют одинаковое (± 1)
число наблюдений, то в качестве метрики выберем `accuracy`, а в качестве функции потерь `binary_crossentropy`.

Обучим модель в течение 50 эпох, с размером батча равным 10 и 10% валидационных данных из тренировочного набора.

## Результаты модели

| ![](https://imgur.com/Vklp5EF.png) |
|:---:|
| График функции потерь |

| ![](https://imgur.com/r3NKiyo.png) |
|:---:|
| График точности |

| ![](https://imgur.com/s6gusUg.png) |
|:---:|
|График результатов классификации|


После обучения получаем следующие показатели на тестовых данных:
```
Epoch 50/50
36/36 [==============================] - 0s 940us/step - loss: 0.0664 - accuracy: 0.9871 - val_loss: 0.0501 - val_accuracy: 1.0000
4/4 [==============================] - 0s 665us/step - loss: 0.0436 - accuracy: 1.0000
[0.04358372092247009, 1.0]
```
