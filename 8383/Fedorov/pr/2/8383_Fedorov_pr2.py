# импорт модулей
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mclr
from tensorflow.keras import layers
from tensorflow.keras import models

#Функцию выбрать в зависимости от варианта (№3)
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
    plt.scatter(data[:, 0], data[:, 1], s=30, c=label[:, 0], cmap=mclr.ListedColormap(['red', 'blue']))     #что получилось
    plt.scatter(data[:, 0], data[:, 1], s=10, c=p_label, cmap=mclr.ListedColormap(['red', 'blue']))             #как правильно
    plt.grid()
    plt.show()


# - 0.934
# 0. 0. 1. 1. 0.
(train_data, train_label), (test_data, test_label) = genData()


#В данном месте необходимо создать модель и обучить ее
model = models.Sequential()
model.add(layers.Dense(16, activation='relu', input_shape=(train_data.shape[1],)))
model.add(layers.Dense(16, activation='relu'))
model.add(layers.Dense(1, activation='sigmoid')) 

model.compile(optimizer='rmsprop', loss='binary_crossentropy', metrics=['accuracy'])  #бинарная кросс-энтропия 

# создание проверочного набора
    
train_size = round(len(train_data) * 0.8)
val_data_x = train_data[train_size:, :]
part_x_train = train_data[:train_size, :]
val_data_y = train_label[train_size:]
part_y_train = train_label[:train_size]


H = model.fit(part_x_train, part_y_train, 
                              epochs=80, 
                              batch_size=50, 
                              validation_data=(val_data_x, val_data_y),
                              verbose=2)

#Получение ошибки и точности в процессе обучения

loss = H.history['loss']
val_loss = H.history['val_loss']
acc = H.history['acc']
val_acc = H.history['val_acc']
epochs = range(1, len(loss) + 1)

#Построение графика ошибки

plt.plot(epochs, loss, 'b.', label='Training loss')
plt.plot(epochs, val_loss, 'b', label='Validation loss')
plt.title('Training and validation loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.show()

#Построение графика точности

plt.clf()
plt.plot(epochs, acc, 'b.', label='Training acc')
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
pred = model.predict(all_data)                              #предсказывание вероятности пренадлежности
drawResults(all_data, all_label, pred)

