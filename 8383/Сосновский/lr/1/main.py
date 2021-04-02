import pandas
import numpy
from tensorflow.keras.layers import Dense
from tensorflow.keras.models import Sequential
from tensorflow.keras.utils import to_categorical
from sklearn.preprocessing import LabelEncoder
import matplotlib.pyplot as plt

epochs = 100

# Загрузка данных
dataframe = pandas.read_csv("iris.csv", header=None)
dataset = dataframe.values
X = dataset[:, 0:4].astype(float)
Y = dataset[:, 4]

# Переход от текстовых меток к категориальному вектору
encoder = LabelEncoder()
encoder.fit(Y)
encoded_Y = encoder.transform(Y)
dummy_y = to_categorical(encoded_Y)

# Создание модели
model = Sequential()
model.add(Dense(4, activation='relu'))
model.add(Dense(24, activation='relu'))
model.add(Dense(3, activation='softmax'))

# Инициализация параметров обучения
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# Обучение сети
history = model.fit(X, dummy_y, epochs=75, batch_size=10, validation_split=0.1, verbose=False)

loss = history.history['loss']
val_loss = history.history['val_loss']
epochs = range(1, len(loss) + 1)
plt.plot(epochs, loss, 'bo', label='Тренировочные ошибки')
plt.plot(epochs, val_loss, 'b', label='Проверочные ошибки')
plt.title('Тренировочные и проверочные ошибки')
plt.xlabel('Эпохи')
plt.ylabel('Ошибки')
plt.legend()
plt.show()

plt.clf()
acc = history.history['accuracy']
val_acc = history.history['val_accuracy']
plt.plot(epochs, acc, 'bo', label='Тренировочная точность')
plt.plot(epochs, val_acc, 'b', label='Проверочная точность')
plt.title('Тренировочная и проверочная точность')
plt.xlabel('Эпохи')
plt.ylabel('Точность')
plt.legend()
plt.show()
