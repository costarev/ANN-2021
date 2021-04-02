# Практическое задание №2, Вариант 5 
Необходимо дополнить фрагмент кода моделью ИНС, которая способна провести бинарную классификацию по сгенерированным данным.
```python
def genData(size=500):
    size1 = size//2
    size2 = size - size1
    x1 = np.random.rand(size1, 1)*1.3 - 0.95
    y1 = np.asarray([3.5*(i+0.2)**2 - 0.8 + (np.random.rand(1)-0.5)/3 for i in x1])
    data1 = np.hstack((x1, y1))
    label1 = np.zeros([size1, 1])
    div1 = round(size1*0.8)
    x2 = np.random.rand(size2, 1)*1.3 - 0.35
    y2 = np.asarray([-3.5*(i-0.2)**2 + 0.8 + (np.random.rand(1)-0.5)/3 for i in x2])
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
    
# Описание решения
Была построена модель из трёх слоёв:

1-ый слой (входной): input_shape = 2

2-ой слой (скрытый): функция relu, количество нейронов 32

3-ий слой (скрытый): фунцкия relu, количество нейронов 32

4-ый слой (выходной): функция sigmoid, количество нейронов 1

Далее модели были установлены следующие параметры обучения:

``optimizer='rmsprop', loss='binary_crossentropy', metrics=['accuracy']``

Затем нейросеть была обучена следующим образом:

``model.fit(train_data, train_label, epochs=70, batch_size=50, validation_data=(test_data, test_label))``

(70 эпох, размер пакета данных 50)

# Результат выполнения работы
В результате работы нейронной сети были получены следующие графики тренировочных и проверочных ошибок и точности:
![](https://cdn1.savepice.ru/uploads/2021/2/25/43605d6c57481a6150d2cad022f0345c-full.png)
![](https://cdn1.savepice.ru/uploads/2021/2/25/8abf5a79a230b1856957a911ed801c61-full.png)

График результатов бинарной классификации:

![](https://cdn1.savepice.ru/uploads/2021/2/25/b2fcfad25b2ee77570a68ed95e1e4803-full.png)

Полученная точность нейросети равна 0.9800000095367432
