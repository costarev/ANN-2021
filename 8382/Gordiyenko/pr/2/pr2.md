<h1>Практика №2

<h2>Задание

<h4>Необходимо дополнить следующий фрагмент кода моделью ИНС, которая способна провести бинарную классификацию по сгенерированным данным:

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

<br><br>
<h2>Вариант 3

<h4> В соответствие с вариантом был взят генератор входных данных
    
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
<br><br>

<h2>Реализация первой модели

<h4> Для первоначальной оценки успехов обучения была выбрана модель с одним промежуточным слоем с функцией 'relu' из 10 нейронов и выходным слоем из одного некрона с функцией 'sigmoid'
<br>В качестве параметров обучения были выбраны: Оптимизатор - rmsprop, функция потерь - binary_crossentropy, метрика - точность, эпохи - 20, размер партии - 10.
<br><br>

<h2>Результат первой модели

<h4> Первая модель показала результат потерь - 0.5078 и точность - 0.6675 на данных для проверки; потери - 0.5096 и точность - 0.7300 на данных для обучения.
<br>
Такие показатели дают понимания, какие корректировки нужны, чтобы повысить точность работы ИНС.
<br><br>

<h2>Реализация второй модели
<h4> Во второй модели было принято решение добавить второй промежуточный слой нейронов с функцией 'relu'. Также было принято решение увеличить количество нейронов в ооих слоях до 55.
<br> В качестве параметров обучения были выбраны: Оптимизатор - rmsprop, функция потерь - binary_crossentropy, метрика - точность, эпохи - 90, размер партии - 10.
<br><br>
<h2>Результат второй модели

<h4> Вторая модель показала результат потерь - 0.0205 и точность - 0.995 на данных для проверки; потери - 0.043 и точность - 0.98 на данных для обучения.
<br>
Такие показатели дают понимания, что параметры обучения ИНС подобраны корректно и ИНС способно точно давать правильный результат.
