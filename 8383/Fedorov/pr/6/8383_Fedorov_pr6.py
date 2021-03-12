import numpy as np
import pandas
import matplotlib.pyplot as plt
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Convolution2D, MaxPooling2D, Dense, Dropout, Flatten
from tensorflow.keras.layers import Dense
from tensorflow.keras.models import Sequential

from tensorflow.keras.utils import to_categorical
from sklearn.preprocessing import LabelEncoder

import var2

# графики потерь и точности при обучении и тестирования
def plot_model_loss_and_accuracy(history, figsize_=(10,5)):
    plt.figure(figsize=figsize_)
    train_loss = history.history['loss']
    test_loss = history.history['val_loss']   
    train_acc = history.history['acc']
    test_acc = history.history['val_acc']    
    epochs = range(1, len(train_loss) + 1)
    
    plt.subplot(121)
    plt.plot(epochs, train_loss, 'r--', label='Training loss')
    plt.plot(epochs, test_loss, 'b-', label='Testing loss')
    plt.title('Graphs of losses during training and testing')
    plt.xlabel('epochs')
    plt.ylabel('loss')
    plt.legend()    
    plt.subplot(122)
    plt.plot(epochs, train_acc, 'r-.', label='Training acc')
    plt.plot(epochs, test_acc, 'b-', label='Testing acc')
    plt.title('Graphs of accuracy during training and testing')
    plt.xlabel('epochs', fontsize=11, color='black')
    plt.ylabel('accuracy', fontsize=11, color='black')
    plt.legend()
    plt.grid(True)
    
    plt.show()


### constants
valid_size = 5
test_size = 5
data_size = 1000
img_size = 50

batch_size = 15
num_epochs = 20
kernel_size = 3         # размер окна
pool_size = 2           # max-pooling
conv_depth_1 = 32 
conv_depth_2 = 64   
drop_prob_1 = 0.25    # dropout
drop_prob_2 = 0.5     # dropout probability


### вектор матриц / вектор слов
data_stack, label_stack = var2.gen_data(size=data_size, img_size=img_size)

print('Data_stack_shape: ')
print(data_stack.shape)
print('Label_stack_shape: ')
print(label_stack.shape)


### перемешивание
idx = np.random.permutation(len(data_stack))
data, labels = data_stack[idx], label_stack[idx]

### перевод меток
encoder = LabelEncoder()
encoder.fit(labels)
labels = encoder.transform(labels)        #[1, 1, 0, 0, 1 ... ]
#labels = to_categorical(encoded_Y)


### разбиение на проверочных, контрольный и тренировочный наборы

test_data = data[:(len(data) // test_size)]
test_data = test_data.reshape((test_data.shape[0], test_data.shape[1], test_data.shape[2], 1))
test_labels = labels[:(len(labels) // test_size)]

data = data[(len(data) // test_size):]
labels = labels[(len(labels) // test_size):]

validation_data = data[:(len(data) // valid_size)]
validation_data = validation_data.reshape((validation_data.shape[0], validation_data.shape[1], validation_data.shape[2], 1))
validation_labels = labels[:(len(labels) // valid_size)]

data = data[(len(data) // valid_size):]
labels = labels[(len(labels) // valid_size):]

train_data = data[:]
train_data = train_data.reshape((train_data.shape[0], train_data.shape[1], train_data.shape[2], 1))
train_labels = labels[:]


### конструирование модели
model = Sequential()
model.add(Convolution2D(conv_depth_1, (kernel_size,kernel_size), activation='relu', input_shape=(img_size,img_size, 1)))
model.add(MaxPooling2D((pool_size,pool_size)))

model.add(Convolution2D(conv_depth_2, (kernel_size,kernel_size), activation='relu'))
model.add(MaxPooling2D((pool_size,pool_size)))

model.add(Convolution2D(conv_depth_2, (kernel_size,kernel_size), activation='relu'))

model.add(Flatten())
model.add(Dropout(0.5))
model.add(Dense(256, activation='relu'))
model.add(Dense(1, activation='sigmoid'))

# компиляция и обучение
model.compile(optimizer='rmsprop', loss='binary_crossentropy', metrics=['accuracy'])
history = model.fit(train_data, train_labels,
          batch_size=batch_size, epochs=num_epochs,
          verbose=1, validation_data=(validation_data, validation_labels))
          
test_loss, test_acc = model.evaluate(test_data, test_labels)
print('Test loss: ', test_loss)
print('Test acc: ', test_acc)

plot_model_loss_and_accuracy(history)
