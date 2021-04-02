import pandas
from tensorflow.keras.layers import Dense
from tensorflow.keras.models import Sequential
from tensorflow.keras.utils import to_categorical
from sklearn.preprocessing import LabelEncoder
import matplotlib.pyplot as plt


def showPlots(history, title):
    acc = history.history['accuracy']
    val_acc = history.history['val_accuracy']
    loss = history.history['loss']
    val_loss = history.history['val_loss']
    epochs = range(1, len(loss) + 1)

    fig, ax = plt.subplots(1, 2)
    ax[0].plot(epochs, acc, 'g', label='Training accuracy')
    ax[0].plot(epochs, val_acc, 'r', label='Validation accuracy')
    ax[0].set_title(title + ' accuracy')
    ax[0].legend()

    ax[1].plot(epochs, loss, 'g', label='Training loss')
    ax[1].plot(epochs, val_loss, 'b', label='Validation loss')
    ax[1].set_title(title + ' loss')
    ax[1].legend()

    fig.set_figwidth(10)
    fig.set_figheight(5)

    plt.show()


def createModel(inputs, countEpochs=75, batchSize=10, validationSplit=0.1, title=''):
    model = Sequential()
    for i in range(0, len(inputs) - 1):
        model.add(Dense(inputs[i], activation='relu'))

    model.add(Dense(inputs[len(inputs) - 1], activation='softmax'))
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    history = model.fit(X, dummy_y, epochs=countEpochs, batch_size=batchSize, validation_split=validationSplit)

    showPlots(history, title)
    return [history.history['val_accuracy'][-1], history.history['val_loss'][-1]]


def testEpochs():
    modelsEpochs = []
    loss = []
    epochs = [75, 150, 225, 300]
    for item in epochs:
        [acc, lss] = createModel([4, 4, 3], item)
        modelsEpochs.append(acc)
        loss.append(lss)

    fig, ax = plt.subplots(1, 2)
    ax[0].plot(epochs, modelsEpochs, 'r')
    ax[0].set_title('Epochs test accuracy')
    ax[0].set_xlabel('epochs')
    ax[0].set_ylabel('values')

    ax[1].plot(epochs, loss, 'b')
    ax[1].set_title('Epochs test loss')
    ax[1].set_xlabel('epochs')


    fig.set_figwidth(10)
    fig.set_figheight(5)

    plt.show()


def testNeurons():
    neurons = [4, 16, 32, 64, 128]

    modelsNeurons3 = []
    lss3 = []
    for item in neurons:
        [ns, ls3] = createModel([4, item, 3], 75)
        modelsNeurons3.append(ns)
        lss3.append(ls3)

    modelsNeurons4 = []
    lss4 = []
    for item in neurons:
        [ns, ls4] = createModel([4, item, item, 3], 75)
        modelsNeurons4.append(ns)
        lss4.append(ls4)

    modelsNeurons5 = []
    lss5 = []
    for item in neurons:
        [ns, ls5] = createModel([4, item, item, item, 3], 75)
        modelsNeurons5.append(ns)
        lss5.append(ls5)

    modelsNeurons6 = []
    lss6 = []
    for item in neurons:
        [ns, ls6] = createModel([4, item, item, item, item, 3], 75)
        modelsNeurons6.append(ns)
        lss6.append(ls6)

    fig, ax = plt.subplots(2, 2)
    ax[0][0].plot(neurons, modelsNeurons3, 'r', label='accuracy')
    ax[0][0].plot(neurons, lss3, 'b', label='loss')
    ax[0][0].legend()
    ax[0][0].set_title('3 layouts')
    ax[0][0].set_xlabel('neurons count')
    ax[0][0].set_ylabel('values')

    ax[0][1].plot(neurons, modelsNeurons4, 'r', label='accuracy')
    ax[0][1].plot(neurons, lss4, 'b', label='loss')
    ax[0][1].legend()
    ax[0][1].set_title('4 layouts')
    ax[0][1].set_xlabel('neurons count')
    ax[0][1].set_ylabel('values')

    ax[1][0].plot(neurons, modelsNeurons5, 'r', label='accuracy')
    ax[1][0].plot(neurons, lss5, 'b', label='loss')
    ax[1][0].legend()
    ax[1][0].set_title('5 layouts')
    ax[1][0].set_xlabel('neurons count')
    ax[1][0].set_ylabel('values')

    ax[1][1].plot(neurons, modelsNeurons6, 'r', label='accuracy')
    ax[1][1].plot(neurons, lss6, 'b', label='loss')
    ax[1][1].legend()
    ax[1][1].set_title('6 layouts')
    ax[1][1].set_xlabel('neurons count')
    ax[1][1].set_ylabel('values')

    fig.set_figwidth(10)
    fig.set_figheight(10)

    plt.show()


def testBatchSize():
    modelsBatchSize = []
    loss = []
    sizes = [10, 20, 30, 40, 50]
    for item in sizes:
        [acc, lss] = createModel([4, 4, 3], 75, item)
        modelsBatchSize.append(acc)
        loss.append(lss)

    fig, ax = plt.subplots(1, 2)
    ax[0].plot(sizes, modelsBatchSize, 'r')
    ax[0].set_title('Batch size test accuracy')
    ax[0].set_xlabel('batch size')
    ax[0].set_ylabel('values')

    ax[1].plot(sizes, loss, 'b')
    ax[1].set_title('Batch size test loss')
    ax[1].set_xlabel('batch size')
    ax[1].set_ylabel('values')

    fig.set_figwidth(10)
    fig.set_figheight(5)

    plt.show()


def testValidationSplit():
    modelsValidationSplit = []
    loss = []
    sizes = [0.1, 0.2, 0.3]
    for item in sizes:
        [acc, lss] = createModel([4, 4, 3], 75, 10, item)
        modelsValidationSplit.append(acc)
        loss.append(lss)

    fig, ax = plt.subplots(1, 2)
    ax[0].plot(sizes, modelsValidationSplit, 'r')
    ax[0].set_title('Validation split test accuracy')
    ax[0].set_xlabel('validation split')
    ax[0].set_ylabel('values')

    ax[1].plot(sizes, loss, 'b')
    ax[1].set_title('Validation split test loss')
    ax[1].set_xlabel('validation split')
    ax[1].set_ylabel('values')

    fig.set_figwidth(10)
    fig.set_figheight(5)

    plt.show()


dataframe = pandas.read_csv("iris.csv", header=None)
dataset = dataframe.values
X = dataset[:, 0:4].astype(float)
Y = dataset[:, 4]

encoder = LabelEncoder()
encoder.fit(Y)
encoder_Y = encoder.transform(Y)
dummy_y = to_categorical(encoder_Y)

#testEpochs()
#testBatchSize()
#testValidationSplit()
#testNeurons()

# model 1
#createModel([4, 4, 4, 3], 255, 10, 0.1, 'model 1')

# model 2 - the best
createModel([4, 32, 32, 3], 255, 10, 0.1, 'model 2')

# model 3
#createModel([4, 64, 64, 3], 255, 10, 0.1, 'model 3')
