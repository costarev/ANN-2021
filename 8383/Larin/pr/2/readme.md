# Практическое задание 2
Ларин Антон  
Гр. 8383
  
## Условие задачи

> Вариант 2 

Необходимо дополнить следующий фрагмент кода моделью ИНС, которая способна провести бинарную классификацию по сгенерированным данным:
```python
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mclr
from tensorflow.keras import layers
from tensorflow.keras import models

#Вариант 2
def genData(size=500):
    data = np.random.rand(size, 2)*2 - 1
    label = np.zeros([size, 1])
    for i, p in enumerate(data):
        if p[0]*p[1] >= 0:
            label[i] = 1.
        else:
            label[i] = 0.
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
Получившаяся модель имеет сделующий вид:
```python
model = models.Sequential()
model.add(layers.Dense(20, activation='relu', input_shape=(2,)))
model.add(layers.Dense(5, activation='relu'))
model.add(layers.Dense(1, activation='sigmoid'))
```
Обучение происходит на 50-ти эпохах. Batch size равен 10

## Характеристики и результаты работы модели

Обучение модели:
```python
H = model.fit(partial_x_train,
                    partial_y_train,
                    epochs=n_epochs,
                    batch_size=int(n_samp/50),
                    validation_data=(x_val, y_val))
```

Точность модели составляет 97-99%

График точности модели:  
![alt text](https://github.com/TxAnton/uni32/raw/ANN_pr2/ANN/prakt/8383/Larin/pr/2/img/Pr2_acc.png)
  
График функции потерь модели:  
![alt text](https://github.com/TxAnton/uni32/raw/ANN_pr2/ANN/prakt/8383/Larin/pr/2/img/Pr2_loss.png)

Последняя строка логов:
```
Epoch 100/100  
17/17 [==============================] - 0s 1ms/step - loss: 0.0624 - accuracy: 0.9888 - val_loss: 0.1319 - val_accuracy: 1.0000
```