import pandas
from tensorflow.keras.layers import Dense
from tensorflow.keras.models import Sequential
from sklearn.preprocessing import LabelEncoder
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn import svm
from sklearn.model_selection import train_test_split

def reduceData(n_components, X):
    pca = PCA(n_components=n_components, svd_solver='full')
    pca.fit(X)
    reduced_X = pca.transform(X)
    return reduced_X

def visualizeData(X,Y):
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.scatter(X[:,0], X[:,1], c=Y,lw=0)
    print(Y)
    fig.show()

def logisticRegression(X,Y):
    clf = LogisticRegression(random_state=0).fit(X, Y)
    print(clf.score(X,Y))

def Svm(X,Y):
    X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size = 0.2)
    clf = svm.SVC(C=5,kernel="rbf",gamma=0.1)
    clf.fit(X_train, y_train)
    print(clf.n_support_)
    print(clf.score(X_train, y_train))
    print(clf.score(X_test,y_test))

def trainModel(X,encoded_Y):
    model = Sequential()
    model.add(Dense(60, input_dim=30, kernel_initializer="normal", bias_initializer="normal", activation='relu'))
    model.add(Dense(15, kernel_initializer="normal", bias_initializer="normal", activation='relu'))
    model.add(Dense(1, kernel_initializer='normal', bias_initializer="normal", activation='sigmoid'))
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    H = model.fit(X, encoded_Y, epochs=100, batch_size=8, validation_split=0.1)
    return H

def plotTrainingGraphs(H):
    loss = H.history['loss']
    val_loss = H.history['val_loss']
    acc = H.history['accuracy']
    val_acc = H.history['val_accuracy']
    epochs = range(1, len(loss) + 1)

    # Построение графика ошибки

    plt.plot(epochs, loss, 'bo', label='Training loss')
    plt.plot(epochs, val_loss, 'b', label='Validation loss')
    plt.title('Training and validation loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.savefig("loss.png")
    plt.show()

    # Построение графика точности

    plt.clf()
    plt.plot(epochs, acc, 'bo', label='Training acc')
    plt.plot(epochs, val_acc, 'b', label='Validation acc')
    plt.title('Training and validation accuracy')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.savefig("acc.png")
    plt.show()


dataframe = pandas.read_csv("sonar.csv", header=None)
dataset = dataframe.values
X = dataset[:,0:60].astype(float)
Y = dataset[:,60]

encoder = LabelEncoder()
encoder.fit(Y)
encoded_Y = encoder.transform(Y)

pca_X = reduceData(30, X)
# visualizeData(pca_X, encoded_Y)
# logisticRegression(X,encoded_Y)
Svm(pca_X,encoded_Y)
# H = trainModel(pca_X, encoded_Y)
# plotTrainingGraphs(H)