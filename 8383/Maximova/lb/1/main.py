import pandas
import matplotlib.pyplot as plt
from tensorflow.keras.layers import Dense
from tensorflow.keras.models import Sequential
from tensorflow.keras.utils import to_categorical
from sklearn.preprocessing import LabelEncoder

dataframe = pandas.read_csv("iris.csv", header=None)
#print(dataframe.head())
dataset = dataframe.values
X = dataset[:, 0:4].astype(float)
Y = dataset[:, 4]

encoder = LabelEncoder()
encoded_Y = encoder.fit_transform(Y)
#print(encoded_Y)
dummy_y = to_categorical(encoded_Y)
#print(dummy_y)

model = Sequential()
model.add(Dense(16, activation='relu'))
model.add(Dense(16, activation='relu'))
model.add(Dense(3, activation='softmax'))
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
hist = model.fit(X, dummy_y, epochs=600, batch_size=5, validation_split=0.1)


#print(hist.history.keys())
#история обратного вызова - получение ошибки и точности в процессе обучения
loss = hist.history['loss']
acc = hist.history['accuracy']
val_loss = hist.history['val_loss']
val_acc = hist.history['val_accuracy']
epochs = range(1, len(loss) + 1)

#Построение графика ошибки
plt.plot(epochs, loss, label='Training loss', linestyle='--', linewidth=2, color='darkmagenta')
plt.plot(epochs, val_loss, 'b', label='Validation loss', color='lawngreen')
plt.title('Training and validation loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.show()

#Построение графика точности
plt.clf()
plt.plot(epochs, acc, label='Training acc', linestyle='--', linewidth=2, color='darkmagenta')
plt.plot(epochs, val_acc, 'b', label='Validation acc', color='lawngreen')
plt.title('Training and validation accuracy')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend()
plt.show()