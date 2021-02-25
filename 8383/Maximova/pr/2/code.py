import numpy as np                                          #большие вычисления с высокой скоростью
import matplotlib.pyplot as plt
import matplotlib.colors as mclr
from tensorflow.keras.layers import Dense
from tensorflow.keras.models import Sequential

def genData(size=500):
    data = np.random.rand(size, 2) * 2 - 1                  #массив случайных чисел размера size x 2
    label = np.zeros([size, 1])                             #массив столбец из нулей на 500 строк - ответы
    for i, p in enumerate(data):
        if (p[0] + 0.2) ** 2 + (0.6 * p[1]) ** 2 >= 0.25:
            label[i] = 0.
        else:
            label[i] = 1.
    div = round(size * 0.8)
    train_data = data[:div, :]
    test_data = data[div:, :]
    train_label = label[:div, :]
    test_label = label[div:, :]
    return (train_data, train_label), (test_data, test_label)

def drawResults(data, label, prediction):
    p_label = np.array([round(x[0]) for x in prediction])
    plt.scatter(data[:, 0], data[:, 1], s=30, c=label[:, 0], cmap=mclr.ListedColormap(['red', 'blue'])) #График разброса
    plt.scatter(data[:, 0], data[:, 1], s=10, c=p_label, cmap=mclr.ListedColormap(['red', 'blue']))
    plt.grid()
    plt.show()

(train_data, train_label), (test_data, test_label) = genData()


#создание последовательной модели
model = Sequential()

#добавление слоев
model.add(Dense(16, activation='relu', input_shape=(2,)))
model.add(Dense(16, activation='relu'))
model.add(Dense(1, activation='sigmoid'))                                                       #sigmoid [0 1]

#настройка обучения модели
model.compile(optimizer='rmsprop', loss='binary_crossentropy', metrics=['accuracy'])

#обучение
hist = model.fit(train_data, train_label, epochs=90, batch_size=15, validation_split=0.2)
#hist - словарь с данными обо всем происходившем во время обучения


#Получение ошибки и точности в процессе обучения
loss = hist.history['loss']
val_loss = hist.history['val_loss']
acc = hist.history['accuracy']
val_acc = hist.history['val_accuracy']
epochs = range(1, len(loss) + 1)

#Построение графика ошибки
plt.plot(epochs, loss, label='Training loss', linestyle='--', linewidth=2, color="darkmagenta")
plt.plot(epochs, val_loss, 'b', label='Validation loss', color="lawngreen")
plt.title('Training and validation loss')                           #оглавление на рисунке
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.show()

#Построение графика точности
plt.clf()
plt.plot(epochs, acc, label='Training acc', linestyle='--', linewidth=2, color="darkmagenta")
plt.plot(epochs, val_acc, 'b', label='Validation acc', color="lawngreen")
plt.title('Training and validation accuracy')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend()
plt.show()

#Получение и вывод результатов на тестовом наборе
results = model.evaluate(test_data, test_label)                     #оценка на тестовых данных
print(results)

#Вывод результатов бинарной классификации
all_data = np.vstack((train_data, test_data))                       #соединяет массивы по вертикали
all_label = np.vstack((train_label, test_label))
pred = model.predict(all_data)                                      #генерирует выходные прогнозы для входных выборок
drawResults(all_data, all_label, pred)