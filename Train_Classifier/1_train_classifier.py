from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV
from sklearn import model_selection
from sklearn import datasets
from sklearn.metrics import accuracy_score
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import cross_val_predict
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
from tensorflow.python.keras.optimizer_v2.adam import Adam
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Conv2D, Dropout, Flatten, MaxPooling2D
from sklearn.preprocessing import OneHotEncoder
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn import svm
from tensorflow.keras.datasets import mnist
import tensorflow as tf
from tensorflow import keras
import numpy as np
import random
#
# gpus = tf.config.experimental.list_physical_devices('GPU')
# print(gpus)
# tf.config.set_visible_devices([], 'CPU') # hide the CPU
# tf.config.set_visible_devices(gpus[0], 'GPU') # unhide potentially hidden GPU
# tf.config.get_visible_devices()

dataset = datasets.load_digits()


image_features = dataset.images.reshape((len(dataset.images), -1))
image_targets = dataset.target

feature_train, feature_test, target_train, target_test = train_test_split(image_features, image_targets, test_size=.2)

k_fold = model_selection.KFold(n_splits=10)


def compute_accuracy(best_model, cross_validate=False):
    print("[Info] {} | Accuracy Score: ".format(best_model), best_model.score(feature_train, target_train))
    best_model.fit(feature_test, target_test)
    y_pred = best_model.predict(feature_test)
    print("[Info] {} | Confusion Matrix: \n".format(best_model),  confusion_matrix(target_test, y_pred))
    if cross_validate:
        predictions = model_selection.cross_val_predict(best_model, feature_test, target_test, cv=k_fold)
        print("Accuracy of the tuned model: {} ".format(best_model), accuracy_score(target_test, predictions))
        # Apply scaling on testing data, without leaking training data.
        predicted = cross_val_predict(best_model, feature_train, target_train, cv=10)
        print("[Info] Cross Validate:  ", predicted)
        print(confusion_matrix(target_train, predicted))
        fig, ax = plt.subplots()
        ax.scatter(target_train, predicted, edgecolors=(0, 0, 0))
        ax.plot([target_test.min(), target_test.max()], [target_test.min(), target_test.max()], "k--", lw=4)
        ax.set_xlabel("Measured")
        ax.set_ylabel("Predicted")
        plt.show()

    img = dataset.images[random.randint(0, 9)]
    print("[Info] Test Image Shape: ", img.shape)
    plt.imshow(img, cmap=plt.cm.gray_r, interpolation='nearest')
    print("[Info] {} | Prediction for test image is : ".format(best_model), best_model.predict(img.reshape(1, -1)))
    plt.show()


def predict_digit_mnist(model, img_path):
    from PIL import Image
    # resize image to 28x28 pixels
    img = Image.open(img_path)
    img = img.resize((8, 8))
    # convert rgb to grayscale
    img = img.convert('L')
    img = np.array(img)
    # reshaping to support our model input and normalizing
    img = img.reshape(1, 8, 8, 1)
    # img = img/255.0
    # predicting the class
    res = model.predict([img])[0]
    return np.argmax(res), max(res)


def train_random_forest_model():
    random_forest_model = RandomForestClassifier(n_jobs=-1, max_features='sqrt')

    param_grid = {
        "n_estimators": [100, 500, 1000],
        "max_depth": [10, 15],
        "min_samples_leaf": [10, 20]
    }
    grid_search = GridSearchCV(estimator=random_forest_model, param_grid=param_grid, cv=10)
    grid_search.fit(feature_train, target_train)
    print('[Info] Best Params: ', grid_search.best_params_)

    optimal_estimators = grid_search.best_params_.get("n_estimators")
    optimal_depth = grid_search.best_params_.get("max_depth")
    optimal_leaf = grid_search.best_params_.get("min_samples_leaf")

    best_model = RandomForestClassifier(n_estimators=optimal_estimators, max_depth=optimal_depth, max_features='sqrt',
                                        min_samples_leaf=optimal_leaf)

    compute_accuracy(best_model)


def train_logistic_regression():
    # Standardization, or mean removal and variance scaling
    pipe = make_pipeline(StandardScaler(), LogisticRegression())
    pipe.fit(feature_train, target_train)  # apply scaling on training data
    # Pipeline(steps=[('standardscaler', StandardScaler()),
    #                 ('logisticregression', LogisticRegression())])
    compute_accuracy(pipe)


def train_svm():
    classifier = svm.SVC(gamma=0.005)
    # Standardization, or mean removal and variance scaling
    pipe = make_pipeline(StandardScaler(), classifier)
    pipe.fit(feature_train, target_train)  # apply scaling on training data
    # Pipeline(steps=[('standardscaler', StandardScaler()),
    #                 ('logisticregression', LogisticRegression())])
    compute_accuracy(pipe)
    # let's test on the last few images


def train_cnn():
    num_classes = 10
    # Hyperparameters
    batch_size = 128
    epochs = 50
    learning_rate = 0.5
    # the data, split between train and test sets
    x_train, y_train, x_test, y_test = feature_train, target_train, feature_test, target_test
    print(x_train.shape, y_train.shape)

    x_train = x_train.reshape(x_train.shape[0], 8, 8, 1)
    x_test = x_test.reshape(x_test.shape[0], 8, 8, 1)
    input_shape = (8, 8, 1)

    # convert class vectors to binary class matrices
    y_train = keras.utils.to_categorical(y_train, num_classes)
    y_test = keras.utils.to_categorical(y_test, num_classes)

    x_train = x_train.astype('float32')
    x_test = x_test.astype('float32')
    # After removing this normalization , I got 0.988 accuracy.
    # x_train /= 255
    # x_test /= 255
    print('x_train shape:', x_train.shape)
    print(x_train.shape[0], 'train samples')
    print(x_test.shape[0], 'test samples')

    model = Sequential()
    model.add(Conv2D(32, kernel_size=(3, 3), activation='relu', input_shape=input_shape))
    model.add(Conv2D(64, (3, 3), activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.25))    # To avoid Overfitting
    model.add(Flatten())
    model.add(Dense(256, activation='relu'))
    model.add(Dropout(0.5))     # To avoid Overfitting
    model.add(Dense(num_classes, activation='softmax'))

    model.compile(loss=keras.losses.categorical_crossentropy,
                  optimizer=keras.optimizers.Adadelta(learning_rate=learning_rate),
                  metrics=['accuracy'])

    hist = model.fit(x_train, y_train, batch_size=batch_size, epochs=epochs, verbose=1,
                     validation_data=(x_test, y_test))
    print(hist)
    results = model.evaluate(x_test, y_test)
    print("[Info] CNN | Loss and Accuracy on the test dataset : ")
    print(results)
    img = dataset.images[random.randint(0, 9)]
    print("[Info] Test Image Shape: ", img.shape)
    plt.imshow(img, cmap=plt.cm.gray_r, interpolation='nearest')
    pred = model.predict(img.reshape(1, -1))
    print("[Info] DNN | Prediction for test image: ", np.argmax(pred[0]))
    plt.show()


def train_deep_neural_network():
    features = dataset.data
    y = dataset.target.reshape(-1, 1)
    print('[Info] Feature shape: ', features.shape)
    encoder = OneHotEncoder(sparse=False)
    targets = encoder.fit_transform(y)

    # Hyperparameter
    epochs = 50
    learning_rate = 0.005
    test_size = 0.2

    train_features, test_features, train_targets, test_targets = train_test_split(features, targets,
                                                                                  test_size=test_size)
    input_dim = int(features.shape[1])
    num_pixels = input_dim
    num_classes = 10
    model = Sequential()
    # first parameter is output dimension
    model.add(Dense(num_pixels, input_dim=input_dim, activation='relu'))
    model.add(Dense(128, input_dim=128, activation='relu'))
    model.add(Dense(128, input_dim=128, activation='relu'))
    model.add(Dense(num_classes, activation='softmax'))

    # we can define the loss function MSE or negative log likelihood
    # optimizer will find the right adjustments for the weights: SGD, Adagrad, ADAM ...
    optimizer = Adam(lr=learning_rate)
    model.compile(loss='binary_crossentropy',
                  optimizer=optimizer,
                  metrics=['accuracy'])

    model.fit(train_features, train_targets, epochs=epochs, batch_size=20, verbose=2)

    # predicted = model.predict(test_features)
    # print('[Info] Prediction of : Actual value: ', predicted.shape, test_targets[0])
    results = model.evaluate(test_features, test_targets)
    print("[Info] DNN | Loss and Accuracy on the test dataset : ")
    print(results)
    # print(confusion_matrix(target_train, predicted))
    # model.save('saved_model/dnn_digit_classifier')
    img = dataset.images[random.randint(0, 9)]
    print("[Info] Test Image Shape: ", img.shape)
    plt.imshow(img, cmap=plt.cm.gray_r, interpolation='nearest')
    pred = model.predict(img.reshape(1, -1))
    print("[Info] DNN | Prediction for test image: ", np.argmax(pred[0]))
    plt.show()


def train_dnn_mnist():
    epochs = 5
    (x_train, y_train), (x_test, y_test) = mnist.load_data()
    print('Training Dataset shape: ', x_train.shape, y_train.shape)
    x_train = tf.keras.utils.normalize(x_train, axis=1)
    x_test = tf.keras.utils.normalize(x_test, axis=1)
    print('Training Dataset shape: ', x_train.shape, y_train.shape)

    model = Sequential()
    model.add(Flatten(input_shape=(28, 28)))
    model.add(Dense(128, activation=tf.nn.relu))
    model.add(Dense(128, activation=tf.nn.relu))
    model.add(Dense(10, activation=tf.nn.softmax))

    model.compile(optimizer='adam',
                  loss='sparse_categorical_crossentropy',
                  metrics=['accuracy']
                  )
    model.fit(x_train, y_train, epochs=epochs)
    model.save('save_model/dnn_mnist_v2')
    results = model.evaluate(x_test, y_test)
    print("[Info] DNN | Loss and Accuracy on the test dataset : ")
    print(results)
    # how to use model
    pred = model.predict(x_test[0].reshape(1, 28, 28, 1))
    print(np.argmax(pred[0]), y_test[0])


def train_deep_neural_network_iris():
    dataset = load_iris()

    features = dataset.data
    y = dataset.target.reshape(-1, 1)

    encoder = OneHotEncoder(sparse=False)
    targets = encoder.fit_transform(y)

    # Hyperparameter
    epochs = 1000
    learning_rate = 0.05
    test_size = 0.2

    train_features, test_features, train_targets, test_targets = train_test_split(features, targets, test_size=test_size)

    model = Sequential()
    # first parameter is output dimension
    model.add(Dense(10, input_dim=4, activation='relu'))
    model.add(Dense(10, input_dim=10, activation='relu'))
    model.add(Dense(10, input_dim=10, activation='relu'))
    model.add(Dense(3, activation='softmax'))

    # we can define the loss function MSE or negative log likelihood
    # optimizer will find the right adjustments for the weights: SGD, Adagrad, ADAM ...
    optimizer = Adam(lr=learning_rate)
    model.compile(loss='categorical_crossentropy',
                  optimizer=optimizer,
                  metrics=['accuracy'])

    model.fit(train_features, train_targets, epochs=epochs, batch_size=20, verbose=2)

    predicted = model.predict(test_features)
    print('[Info] Prediction of : Actual value: ', predicted.shape, test_targets[0])
    results = model.evaluate(test_features, test_targets)
    print("[Info] Loss and Accuracy on the test dataset : ")
    print(results)
    # print(confusion_matrix(target_train, predicted))


def show_dataset():
    print(dataset.data.shape)
    print(dataset.images[0].flatten().shape)
    plt.matshow(dataset.images[0])
    plt.show()


train_logistic_regression()
train_random_forest_model()
train_deep_neural_network_iris()
train_deep_neural_network()
train_dnn_mnist()
train_svm()
train_cnn()