import pandas
import matplotlib.pyplot as plt
from tensorflow.keras.layers import Dense #полносвязанный слой
from tensorflow.keras.models import Sequential #сеть прямого распространения
from sklearn.preprocessing import LabelEncoder #для кодирования из строк в целочисленныые значения

dataframe = pandas.read_csv("sonar.csv", header=None)
dataset = dataframe.values
X = dataset[:, 0:30].astype(float)
Y = dataset[:, 60]

encoder = LabelEncoder()
encoder_Y = encoder.fit_transform(Y) #преобразовать в числа и вернуть результат
#print(encoder_Y)

model = Sequential()
model.add(Dense(60, input_dim=60, kernel_initializer='normal', activation='relu')) #init->kernel_initializer в новом tensorflow
model.add(Dense(15, kernel_initializer='normal', activation='relu'))
model.add(Dense(15, kernel_initializer='normal', activation='relu'))
model.add(Dense(1, kernel_initializer='normal', activation='sigmoid'))
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
hist = model.fit(X, encoder_Y, epochs=100, batch_size=10, validation_split=0.1)

loss = hist.history['loss']
acc = hist.history['accuracy']
val_loss = hist.history['val_loss']
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