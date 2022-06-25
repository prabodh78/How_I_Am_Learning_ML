from skimage import data, feature, transform
import matplotlib.pyplot as plt
from sklearn.svm import LinearSVC
from skimage.io import imread
from itertools import chain
from sklearn.datasets import fetch_lfw_people
from sklearn.feature_extraction.image import PatchExtractor
import numpy as np
from sklearn.metrics import accuracy_score, confusion_matrix
from tensorflow import keras
from sklearn.model_selection import train_test_split
from tensorflow.keras.layers import Dense, Conv2D, Dropout, Flatten, MaxPooling2D
from sklearn.preprocessing import OneHotEncoder
from tensorflow.keras.models import Sequential


# we will use PatchExtractor to generate several variants of these images
def generate_random_samples(image, num_of_generated_images=100, patch_size=None):
    extractor = PatchExtractor(patch_size=patch_size, max_patches=num_of_generated_images, random_state=42)
    patches = extractor.transform((image[np.newaxis]))
    return patches


def get_prediction(image_path, model):
    import cv2
    img = cv2.imread(image_path, 0)
    img = transform.resize(img, (62, 47))
    print("[Info] Test Image Shape: ", img.shape)
    plt.imshow(img, cmap=plt.cm.gray_r, interpolation='nearest')
    plt.show()
    prediction = model.predict(img.reshape(1, 62, 47, 1)).argmax()
    print("[Info] CNN | Prediction for given image is : ", prediction)
    return int(prediction)


def train_cnn_face_classifier():
    # we can load a data-set of human faces (positive samples)
    human_faces = fetch_lfw_people()
    positive_images = human_faces.images[:10000]

    # fetch a data-set without faces (negative samples)
    non_face_topics = ['moon', 'text', 'coins']

    negative_samples = [(getattr(data, name)()) for name in non_face_topics]

    # we generate 3000 samples (negative samples without a human face)
    negative_images = np.vstack([generate_random_samples(im, 1000, patch_size=positive_images[0].shape)
                                 for im in negative_samples])

    # Need to appy feature extractor
    image_features = np.concatenate((positive_images, negative_images), axis=0)
    image_targets = np.concatenate((np.ones(len(positive_images)), np.zeros(len(negative_images))), axis=0)
    print("[Info] Dataset Shape : ", image_features.shape, image_targets.shape)
    x_train, x_test, y_train, y_test = train_test_split(image_features, image_targets,
                                                                              test_size=.2)
    print(x_train.shape, y_train.shape)
    batch_size = 128
    num_classes = 10
    epochs = 3

    x_train = x_train.reshape(x_train.shape[0], 62, 47, 1)
    x_test = x_test.reshape(x_test.shape[0], 62, 47, 1)
    input_shape = (62, 47, 1)

    # convert class vectors to binary class matrices
    y_train = keras.utils.to_categorical(y_train, num_classes)
    y_test = keras.utils.to_categorical(y_test, num_classes)

    x_train = x_train.astype('float32')
    x_test = x_test.astype('float32')
    x_train /= 255
    x_test /= 255
    print('x_train shape:', x_train.shape)
    print(x_train.shape[0], 'train samples')
    print(x_test.shape[0], 'test samples')

    model = Sequential()
    model.add(Conv2D(32, kernel_size=(3, 3), activation='relu', input_shape=input_shape))
    model.add(Conv2D(64, (3, 3), activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.25))
    model.add(Flatten())
    model.add(Dense(256, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(num_classes, activation='softmax'))

    model.compile(loss=keras.losses.categorical_crossentropy, optimizer=keras.optimizers.Adadelta(),
                  metrics=['accuracy'])

    hist = model.fit(x_train, y_train, batch_size=batch_size, epochs=epochs, verbose=1,
                     validation_data=(x_test, y_test))
    print("The model has successfully trained")
    model.save('saved_model/cnn_face_detection')
    print("Saving the model")
    print(model.evaluate(x_test, y_test))


# train_cnn_face_classifier()
model = keras.models.load_model('saved_model/cnn_face_detection')
a = get_prediction('../Dataset/personal_images/person_1.jpeg', model)
print(a)


