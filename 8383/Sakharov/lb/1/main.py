#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import pandas
import matplotlib.pyplot as plt
from tensorflow import keras
from tensorflow.keras.layers import Dense
from tensorflow.keras.models import Sequential
from tensorflow.keras.utils import to_categorical
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split


# In[ ]:


dataframe = pandas.read_csv("iris.csv", header=None)
dataset = dataframe.values
X = dataset[:,0:4].astype(float)
Y = dataset[:,4]


# In[ ]:


encoder = LabelEncoder()
encoder.fit(Y)
encoded_Y = encoder.transform(Y)
dummy_y = to_categorical(encoded_Y)


# In[ ]:


def model_0_layers():
    model = Sequential()
    model.add(Dense(4, activation="relu"))
    model.add(Dense(3, activation="softmax"))
    model.compile(optimizer='adam', loss=keras.losses.CategoricalCrossentropy(), metrics=['accuracy'])
    print("Model: without hidden layers")
    return model

def model_1_layers(neurons_count_1):
    model = Sequential()
    model.add(Dense(4, activation="relu"))
    model.add(Dense(neurons_count_1, activation="relu"))
    model.add(Dense(3, activation="softmax"))
    model.compile(optimizer='adam', loss=keras.losses.CategoricalCrossentropy(), metrics=['accuracy'])
    print("Model: layer1:{};".format(neurons_count_1))
    return model

def model_2_layers(neurons_count_1, neurons_count_2):
    model = Sequential()
    model.add(Dense(4, activation="relu"))
    model.add(Dense(neurons_count_1, activation="relu"))
    model.add(Dense(neurons_count_2, activation="relu"))
    model.add(Dense(3, activation="softmax"))
    model.compile(optimizer='adam', loss=keras.losses.CategoricalCrossentropy(), metrics=['accuracy'])
    print("Model: layer1:{}; layer2:{};".format(neurons_count_1, neurons_count_2))
    return model
    
def model_3_layers(neurons_count_1, neurons_count_2, neurons_count_3):
    model = Sequential()
    model.add(Dense(4, activation="relu"))
    model.add(Dense(neurons_count_1, activation="relu"))
    model.add(Dense(neurons_count_2, activation="relu"))
    model.add(Dense(neurons_count_3, activation="relu"))
    model.add(Dense(3, activation="softmax"))
    model.compile(optimizer='adam', loss=keras.losses.CategoricalCrossentropy(), metrics=['accuracy'])
    print("Model: layer1:{}; layer2:{}; layer3:{};".format(neurons_count_1, neurons_count_2, neurons_count_3))
    return model

def fit_model(model, ep=70, bs=10, vs=0.1):
    print("Epochs: {}; Batch size: {}; Validation split: {}".format(ep, bs, vs))
    return model.fit(X, dummy_y, epochs=ep, batch_size=bs, validation_split=vs, verbose=False)

def plot_history(history, filename=""):
    loss = history['loss']
    val_loss = history['val_loss']
    acc = history['accuracy']
    val_acc = history['val_accuracy']
    epochs = range(1, len(loss) + 1)
    fig, (ax1,ax2) = plt.subplots(1, 2, figsize=(18,10))
    fig.suptitle('Loss and accuracy')

    ax1.plot(epochs, loss, color="green", label='Training loss')
    ax1.plot(epochs, val_loss, color = "blue", label='Validation loss')
    ax1.legend()

    ax2.plot(epochs, acc, color="green", label='Training acc')
    ax2.plot(epochs, val_acc, color = "blue", label='Validation acc')
    ax2.legend()

    plt.show()

    if (filename != ""):
        plt.savefig(filename)

def print_history(history):
    loss = history['loss']
    val_loss = history['val_loss']
    acc = history['accuracy']
    val_acc = history['val_accuracy']

    print("T loss: {}; V loss: {}; T accuracy: {}; V accuracy: {}"
          .format(loss[-1], val_loss[-1], acc[-1], val_acc[-1]))
    plot_history(history)
    print('----------------------------------------------------------------------------')
    #plt.show()
    


# In[ ]:


# for ep in [30, 70, 150]:
#     for bs in [1, 3, 10]:
#         for vs in [0.001, 0.1, 0.5]:
#             print("!!! Epochs: {}; Batch size: {} Validation split: {}".format(ep, bs, vs))
#             print_history(fit_model(model_0_layers(), ep, bs, vs).history)
#             print_history(fit_model(model_1_layers(3), ep, bs, vs).history)
#             print_history(fit_model(model_1_layers(5), ep, bs, vs).history)
#             print_history(fit_model(model_1_layers(10), ep, bs, vs).history)
#             print_history(fit_model(model_1_layers(15), ep, bs, vs).history)
#             print_history(fit_model(model_2_layers(3, 5), ep, bs, vs).history)
#             print_history(fit_model(model_2_layers(5, 3), ep, bs, vs).history)
#             print_history(fit_model(model_2_layers(10, 5), ep, bs, vs).history)
#             print_history(fit_model(model_2_layers(5, 10), ep, bs, vs).history)
#             print_history(fit_model(model_2_layers(10, 10), ep, bs, vs).history)
#             print_history(fit_model(model_3_layers(3, 3, 3), ep, bs, vs).history)
#             print_history(fit_model(model_3_layers(10, 10, 10), ep, bs, vs).history)
#             print_history(fit_model(model_3_layers(3, 5, 10), ep, bs, vs).history)
#             print_history(fit_model(model_3_layers(10, 5, 3), ep, bs, vs).history)


# for n1 in range(5, 50, 10):
#     for n2 in range(5, 50, 10):
#         m = model_2_layers(n1, n2)
#         for epochs in range(10, 200, 50):
#             for batch_size in range(5, 20, 5):
#                 #for validation_split in range(0.01, 500, 50):
#                 print("Model: layer1:{} ; layer2:{}".format(n1, n2))
#                 m.fit(X, dummy_y, epochs=10, batch_size=5, validation_split=0.1, verbose=False)
#                 result = fit_model(m, epochs, batch_size, 0.1)
#                 print_history(result.history)


# In[ ]:


model = Sequential()
model.add(Dense(4, activation="relu"))
model.add(Dense(3, activation="softmax"))
model.compile(optimizer='adam', loss=keras.losses.CategoricalCrossentropy(), metrics=['accuracy'])
print_history(fit_model(model, 75, 10, 0.1).history)


# In[ ]:


model = Sequential()
model.add(Dense(4, activation="relu"))
model.add(Dense(5, activation="relu"))
model.add(Dense(3, activation="softmax"))
model.compile(optimizer='adam', loss=keras.losses.CategoricalCrossentropy(), metrics=['accuracy'])
print_history(fit_model(model, 75, 10, 0.1).history)


# In[ ]:


model = Sequential()
model.add(Dense(4, activation="relu"))
model.add(Dense(10, activation="relu"))
model.add(Dense(10, activation="relu"))
model.add(Dense(3, activation="softmax"))
model.compile(optimizer='adam', loss=keras.losses.CategoricalCrossentropy(), metrics=['accuracy'])
print_history(fit_model(model, 75, 10, 0.1).history)


# In[ ]:


model = Sequential()
model.add(Dense(4, activation="relu"))
model.add(Dense(5, activation="relu"))
model.add(Dense(3, activation="softmax"))
model.compile(optimizer='adam', loss=keras.losses.CategoricalCrossentropy(), metrics=['accuracy'])
print_history(fit_model(model, 80, 5, 0.1).history)


# In[ ]:


model = Sequential()
model.add(Dense(4, activation="relu"))
model.add(Dense(8, activation="relu"))
model.add(Dense(3, activation="softmax"))
model.compile(optimizer='adam', loss=keras.losses.CategoricalCrossentropy(), metrics=['accuracy'])
print_history(fit_model(model, 80, 15, 0.1).history)


# In[ ]:




