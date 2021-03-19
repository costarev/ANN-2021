import numpy as np


from tensorflow.python.keras.callbacks import History
from tensorflow.keras.layers import Dense
from tensorflow.keras.models import Sequential
# from tensorflow.keras.utils import to_categorical
from tensorflow.keras.datasets import boston_housing
import matplotlib.pyplot as plt

def build_model():
    model = Sequential()
    model.add(Dense(64, activation='relu', input_shape=(train_data.shape[1],)))
    model.add(Dense(64, activation='relu'))
    model.add(Dense(1))
    model.compile(optimizer='rmsprop', loss='mse', metrics=['mae'])
    return model

def plot_vals(hists, num_epochs: int, is_maes:bool):
    clr = "rgbmcyrgbmcyrgbmcy"
    offset = 0
    # print(hist[0].history.keys())

    if(is_maes):
        vl = 'mae'
        val_vl='val_mae'
    else:
        vl = 'loss'
        val_vl = 'val_loss'
    plt.xlabel("epochs")
    plt.ylabel(f"{vl}")
    lg = []
    for n,i in enumerate(hists):
        print(n,i)

        plt.plot(range(offset,num_epochs), i.history[vl][offset:], f"{clr[n]}.", label=f'Training {vl} {n}')
        plt.plot(range(offset,num_epochs), i.history[val_vl][offset:], f"{clr[n]}", label=f'Validation {vl} {n}')
        lg.append(f'Training {vl} {n}')
        lg.append(f'Validation {vl} {n}')
    plt.legend(lg)
    plt.xlabel("epochs")
    plt.ylabel(f"{vl}")
    plt.show()

def plot_avg_vals(hists, num_epochs: int, is_maes:bool):
    clr = "rgbmcyrgbmcyrgbmcy"
    offset = 0
    # print(hist[0].history.keys())

    if(is_maes):
        vl = 'mae'
        val_vl='val_mae'
    else:
        vl = 'loss'
        val_vl = 'val_loss'
    plt.xlabel("epochs")
    plt.ylabel(f"{vl}")
    lg = []

    vls = []
    val_vls = []

    for i in range(num_epochs):
        vls.append(sum([j.history[vl][i] for j in hists])/len(hists))
        val_vls.append(sum([j.history[val_vl][i] for j in hists])/len(hists))

    plt.plot(range(offset, num_epochs), vls[offset:], "b", label=f'Avg training {vl}')
    plt.plot(range(offset, num_epochs), val_vls[offset:], "r", label=f'Avg validation {vl}')
    lg.append(f'Avg training {vl}')
    lg.append(f'Avg validation {vl}')
    # for n,i in enumerate(hists):
    #     print(n,i)
    #
    #     plt.plot(range(offset,num_epochs), i.history[vl][offset:], f"{clr[n]}.", label=f'Training {vl} {n}')
    #     plt.plot(range(offset,num_epochs), i.history[val_vl][offset:], f"{clr[n]}", label=f'Validation {vl} {n}')
    #     lg.append(f'Training {vl} {n}')
    #     lg.append(f'Validation {vl} {n}')
    plt.legend(lg)
    plt.xlabel("epochs")
    plt.ylabel(f"{vl}")
    print(f"min {vl}: ",min(vls), vls.index(min(vls)))
    plt.show()
# def plot_av_maes


if __name__ == '__main__':
    (train_data, train_targets), (test_data, test_targets) = boston_housing.load_data()

    print(train_data.shape)
    print(test_data.shape)

    print(test_targets)

    mean = train_data.mean(axis=0)
    train_data -= mean
    std = train_data.std(axis=0)
    train_data /= std

    test_data -= mean
    test_data /= std

    k = 6
    num_val_samples = len(train_data) // k

    num_epochs = 40
    all_scores = []
    hists = []
    maes=[] # {"trn":[],"val":[]}
    print("working...")
    for i in range(k):
        # print('processing fold #', i)
        val_data = train_data[i * num_val_samples: (i + 1) * num_val_samples]
        val_targets = train_targets[i * num_val_samples: (i + 1) * num_val_samples]
        partial_train_data = np.concatenate([train_data[:i * num_val_samples], train_data[(i + 1) * num_val_samples:]], axis=0)
        partial_train_targets = np.concatenate(
            [train_targets[:i * num_val_samples], train_targets[(i + 1) * num_val_samples:]], axis=0)
        model = build_model()
        H = model.fit(partial_train_data, partial_train_targets, epochs=num_epochs, batch_size=1,   validation_data=(val_data, val_targets), verbose=0)
        val_mse, val_mae = model.evaluate(val_data, val_targets, verbose=0)
        all_scores.append(val_mae)
        hists.append(H)
        # maes.append({"trn":H.history["mae"],"val":H.history["val_mae"]})
        # maes["trn"].append(H.history["mae"])
        # maes["val"].append(val_mae)
    # print(H)
    # print(list(H))
    print(num_epochs,np.mean(all_scores))
    plot_vals(hists,num_epochs,is_maes=True)
    plot_vals(hists, num_epochs, is_maes=False)

    plot_avg_vals(hists, num_epochs, is_maes=True)
    plot_avg_vals(hists, num_epochs, is_maes=False)
