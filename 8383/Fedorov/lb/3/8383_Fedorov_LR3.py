import pandas
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.layers import Dense
from tensorflow.keras.models import Sequential
from tensorflow.keras.utils import to_categorical
from sklearn.preprocessing import LabelEncoder
from tensorflow.keras.utils import plot_model

#lr3
from tensorflow.keras.datasets import boston_housing

 # загрузка обучающих и контрольных образцов
(train_data, train_targets), (test_data, test_targets) = boston_housing.load_data() 

#нормализация данных
# - среднее и / на ско
mean = train_data.mean(axis=0) 
train_data -= mean
std = train_data.std(axis=0)
train_data /= std

test_data -= mean
test_data /= std


def build_model():
    model = Sequential()
    model.add(Dense(64, activation='relu', input_shape=(train_data.shape[1],)))
    model.add(Dense(64, activation='relu'))
    model.add(Dense(1))
    model.compile(optimizer='rmsprop', loss='mse', metrics=['mae'])
    return model



def plot_model_loss_and_mae(history, index=0, figsize_=(10,5)):
    plt.figure(figsize=figsize_)
    train_loss = history.history['loss']
    val_loss = history.history['val_loss']   
    train_mae = history.history['mean_absolute_error']
    val_mae = history.history['val_mean_absolute_error']    
    epochs = range(1, len(train_loss) + 1)
    plt.ylim(0, 6)
    plt.yticks(np.arange(0,6.5,0.5))
    plt.plot(epochs, train_mae, 'r-.', label='Training mae')
    plt.ylim(0, 6)
    plt.yticks(np.arange(0,6.5,0.5))
    plt.plot(epochs, val_mae, 'b-', label='Testing mae')
    plt.title('Graphs of mae during training and testing')
    plt.xlabel('epochs', fontsize=11, color='black')
    plt.ylabel('mae', fontsize=11, color='black')
    plt.legend()
    plt.grid(True)
    
    name = 'figure' + str(index) + '.png'
    plt.savefig(name)
    #plt.show()

#экспоненциальное скользящее среднее
def smooth_curve_mae(points, alpha=0.9):
    sm_points = []
    for pt in points:
        if sm_points:
            prev = sm_points[-1]
            sm_points.append(prev * alpha + pt * (1 - alpha))
        else:
            sm_points.append(pt)
    return sm_points

 
def plot_average_mae_history(mae_history, figsize_=(10,5), skip_size=10, name = 'smooth_graph.png'):
    avg_mae_history = [np.mean([x[i] for x in mae_history])  for i in range(len(mae_history[0]))]
    plt.figure(figsize=figsize_)
    epochs = range(1, len(avg_mae_history) + 1)
    plt.subplot(121)
    plt.plot(epochs, avg_mae_history, 'b-' , label='Avg mae' )
    plt.title('Average MAE on epochs')  
    plt.xlabel('Epoch')
    plt.ylabel('Mean Abs Error [1000$]')
    plt.legend()
    plt.subplot(122)
    sm_points = smooth_curve_mae(avg_mae_history[skip_size:])
    plt.plot(range(skip_size+1, len(sm_points) + skip_size+ 1), sm_points,  'b-')
    plt.title('Smooth curve avg MAE on epochs')  
    plt.xlabel('Epoch')
    plt.ylabel('Mean Abs Error [1000$]')
    plt.legend()
    plt.grid(True)
    plt.savefig(name)

#перекрестная проверка по К блокам
k = 4
num_val_samples = len(train_data) // k
num_epochs = 60
val_mae_scores = []
val_mae_histories = []


for i in range(4):
    print('Processing fold #', i) 
    # подготовка проверочных данных
    val_data = train_data[i * num_val_samples: (i + 1) * num_val_samples]
    val_targets = train_targets[i * num_val_samples: (i + 1) * num_val_samples]

    # подготовка обучающих данных 
    partial_train_data = np.concatenate([train_data[:i * num_val_samples], 
                                                               train_data[(i + 1) * num_val_samples: ]], 
                                                               axis=0)

    partial_train_targets = np.concatenate([train_targets[:i * num_val_samples], 
                                                               train_targets[(i + 1) * num_val_samples: ]], 
                                                               axis=0)  

    model = build_model()
    history = model.fit(partial_train_data, partial_train_targets,
                                  validation_data=(val_data, val_targets),
                                  epochs=num_epochs, batch_size=1, verbose=0)
    
    # keys() ['loss', 'mean_absolute_error', 'val_loss', 'val_mean_absolute_error']
    val_mae_histories.append(history.history['val_mean_absolute_error'])
    plot_model_loss_and_mae(history, i)
    
    val_mse, val_mae = model.evaluate(val_data, val_targets, verbose=0)
    val_mae_scores.append(val_mae)
 
print('Mean of mae: ', np.mean(val_mae_scores))
print(val_mae_scores)
plot_average_mae_history(val_mae_histories)


#обучить на всех данных и проверить на контрольном наборе
model = build_model()
model.fit(train_data, train_targets, epochs=num_epochs, batch_size=1, verbose=0)

test_mse_score, test_mae_score = model.evaluate(test_data, test_targets)
print('Test MSE score:' , test_mse_score)
print('Test MAE score:', test_mae_score)
