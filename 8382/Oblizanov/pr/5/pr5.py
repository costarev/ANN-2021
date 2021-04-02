import csv

import numpy as np
from keras.layers import Input, Dense
from keras.models import Model
import matplotlib.pyplot as plt

train_size = 500
test_size = 50
epochs_regr = 70
batch_size_regr = 20


def gen_data(size):
    data = []
    labels = []
    for i in range(size):
        x = np.random.normal(0, 10)
        e = np.random.normal(0, 0.3)
        data.append((x ** 2 + x + e, np.sin(x - np.pi / 4) + e, np.log(np.fabs(x)) + e, -1 * x ** 3 + e, -1 * x / 4 + e,
                     -1 * x + e))
        labels.append((np.fabs(x) + e))
    return np.array(data), np.array(labels)


def write_csv(data, name):
    file = open(name, 'w', newline='')
    writer = csv.writer(file, delimiter=',', quoting=csv.QUOTE_MINIMAL)
    for i in data.tolist():
        if type(i) == float:
            writer.writerow([i])
        else:
            writer.writerow(i)


train_data, train_labels = gen_data(train_size)
test_data, test_labels = gen_data(test_size)
write_csv(train_data, 'train_data.csv')
write_csv(train_labels, 'train_labels.csv')
write_csv(test_data, 'test_data.csv')
write_csv(test_labels, 'test_labels.csv')

# data normalization
mean = np.mean(train_data, axis=0)
train_data -= mean
std = np.std(train_data, axis=0)
train_data /= std

test_data -= mean
test_data /= std

write_csv(test_data, 'test_data_normalized.csv')

# encoder model
model_input = Input(shape=(6,), name='model_input')
encoding = Dense(48, activation='relu')(model_input)
encoding = Dense(24, activation='relu')(encoding)
encoding = Dense(4, name='encoded_output')(encoding)

# decoder model (for auto encoder)
decoding = Dense(24, activation='relu', name='dec1')(encoding)
decoding = Dense(48, activation='relu', name='dec2')(decoding)
decoding = Dense(6, name='decoded_output')(decoding)

# regression model
regression = Dense(30, activation='relu')(encoding)
regression = Dense(30, activation='relu')(regression)
regression = Dense(30, activation='relu')(regression)
regression = Dense(1, name='predicted_output')(regression)

double_model = Model(model_input, outputs=[decoding, regression])

# init and learn
double_model.compile(optimizer='adam', loss='mse', loss_weights=[0.8, 0.2])
res_regr = double_model.fit(
    {'model_input': train_data}, {'decoded_output': train_data, 'predicted_output': train_labels}, epochs=epochs_regr,
    batch_size=batch_size_regr,
    validation_data=({'model_input': test_data}, {'decoded_output': test_data, 'predicted_output': test_labels}))

# creating models
regression_model = Model(model_input, regression)
encoder = Model(model_input, encoding)

# creating standalone decoder model
decoder_input = Input(shape=(4,))
decoder_st = double_model.get_layer('dec1')(decoder_input)
decoder_st = double_model.get_layer('dec2')(decoder_st)
decoder_st = double_model.get_layer('decoded_output')(decoder_st)
decoder = Model(decoder_input, decoder_st)

encoder_prediction = encoder.predict(test_data)
decoder_prediction = decoder.predict(encoder_prediction)
regression_prediction = regression_model.predict(test_data)

write_csv(encoder_prediction, 'encoder.csv')
write_csv(regression_prediction, 'regression.csv')
write_csv(decoder_prediction, 'decoder.csv')
write_csv(decoder_prediction * std + mean, 'decoder_denormalized.csv')
encoder.save('encoder_model.h5')
decoder.save('decoder_model.h5')
regression_model.save('regression_model.h5')
