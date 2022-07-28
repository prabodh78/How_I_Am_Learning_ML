import glob
import os

import cv2
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
from tensorflow.keras.layers import Dense, Conv2D, Dropout, Flatten, MaxPooling2D, Rescaling
from sklearn.preprocessing import OneHotEncoder
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn import svm
from tensorflow.keras.datasets import mnist
import tensorflow as tf
from tensorflow import keras
import numpy as np
import random
import dlib


def train_deep_neural_network(dataset):
    classes = {'frontal': 0, 'left': 1, 'right': 2}
    features = []
    images = []
    labels = []
    for key, values in classes.items():
        images = images + glob.glob(os.path.join(dataset, key, '*.jpg'))
        for img_path in glob.glob(os.path.join(dataset, key, '*.jpg')):
            # Todo: Extract Face as a feature
            features.append(cv2.imread(img_path))
            labels.append(values)
    # y = np.array(labels).reshape(-1, 1)
    # features = np.array(features).reshape(-1, 1)
    print('[Info] Feature shape: {} and Labels: {}'.format(features.shape, len(y)))
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
    num_classes = 3

    model = Sequential()
    # first parameter is output dimension
    model.add(Dense(num_pixels, input_dim=input_dim, activation='relu'))
    model.add(Dense(128, input_dim=128, activation='relu'))
    model.add(Dense(128, input_dim=128, activation='relu'))
    model.add(Dense(num_classes, activation='softmax'))

    # we can define the loss function MSE or negative log likelihood
    # optimizer will find the right adjustments for the weights: SGD, Adagrad, ADAM ...
    optimizer = Adam(lr=learning_rate)
    model.compile(loss='categorical_crossentropy',
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
    img = images[random.randint(0, 9)]
    print("[Info] Test Image Shape: ", img.shape)
    plt.imshow(img, cmap=plt.cm.gray_r, interpolation='nearest')
    pred = model.predict(img.reshape(1, -1))
    print("[Info] DNN | Prediction for test image: ", np.argmax(pred[0]))
    plt.show()


def train_cnn(data_dir, debug=True, model_name='face_pose_classifier'):
    batch_size = 32
    img_height = 70
    img_width = 70
    num_classes = len(os.listdir(data_dir))
    # Hyper-parameters
    epochs = 15

    print('[Info] No. of classes: ', num_classes)
    train_ds = tf.keras.utils.image_dataset_from_directory(
        data_dir,
        validation_split=0.2,
        subset="training",
        seed=123,
        image_size=(img_height, img_width),
        batch_size=batch_size)
    val_ds = tf.keras.utils.image_dataset_from_directory(
        data_dir,
        validation_split=0.2,
        subset="validation",
        seed=123,
        image_size=(img_height, img_width),
        batch_size=batch_size)

    class_names = train_ds.class_names
    print(class_names)
    if debug:
        plt.figure(figsize=(10, 10))
        for images, labels in train_ds.take(1):
            for i in range(9):
                ax = plt.subplot(3, 3, i + 1)
                plt.imshow(images[i].numpy().astype("uint8"))
                plt.title(class_names[labels[i]])
                plt.axis("off")
        plt.show()

    for image_batch, labels_batch in train_ds:
        print('Image Shape: ', image_batch.shape)
        print('Label Shape: ', labels_batch.shape)
        break

    normalization_layer = tf.keras.layers.Rescaling(1. / 255)

    normalized_ds = train_ds.map(lambda x, y: (normalization_layer(x), y))
    image_batch, labels_batch = next(iter(normalized_ds))
    first_image = image_batch[0]
    # Notice the pixel values are now in `[0,1]`.
    print(np.min(first_image), np.max(first_image))

    model = Sequential([
        Rescaling(1. / 255, input_shape=(img_height, img_width, 3)),
        Conv2D(16, 3, padding='same', activation='relu'),
        MaxPooling2D(),
        Conv2D(32, 3, padding='same', activation='relu'),
        MaxPooling2D(),
        Conv2D(64, 3, padding='same', activation='relu'),
        MaxPooling2D(),
        Flatten(),
        Dense(128, activation='relu'),
        Dropout(0.2),
        Dense(num_classes, activation='softmax')
    ])
    model.compile(
        optimizer='adam',
        loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
        metrics=['accuracy'])
    model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=epochs
    )
    model.save(model_name)


def preprocess(img):
    face_list = []
    scores_list = []
    src = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    src = cv2.equalizeHist(src)
    hog = dlib.get_frontal_face_detector()
    dets, scores, idx = hog.run(src, 0)
    face = None
    if not dets:
        cnn = dlib.cnn_face_detection_model_v1('mmod_human_face_detector.dat')
        dets = cnn(src, 0)

        for i, d in enumerate(dets):
            if d.confidence > 0.4:
                left = d.rect.left()
                top = d.rect.top()
                if left < 0:
                    left = 0
                if top < 0:
                    top = 0
                face_list.append([left, top, d.rect.right(), d.rect.bottom()])
                scores_list.append(dets[i].confidence)
    else:
        for i, d in enumerate(dets):

            left = d.left()
            top = d.top()
            if left < 0:
                left = 0
            if top < 0:
                top = 0
            face_list.append([left, top, d.right(), d.bottom()])
            if idx[i] == 0.0:
                idx[i] = 3.0
            scores_list.append(scores[i])

        if True:
            # Apply NMS
            indices = cv2.dnn.NMSBoxes(face_list, scores_list, 0.0, 0.3)
            rects_f = []
            scores_f = []
            for idx in indices:
                index = idx[0]
                rects_f.append(face_list[index])
                scores_f.append(scores_list[index])
            face_list = rects_f
            scores_list = scores_f
    if face_list:
        x_min, y_min, x_max, y_max = face_list[0]
        face = img[y_min:y_max, x_min:x_max]
        face = cv2.resize(face, (70, 70), interpolation=cv2.INTER_NEAREST)
        print(face.shape)
    return face


def predict(model, image_numpy):
    face_img = preprocess(image_numpy)
    if face_img is not None:
        # cv2.imshow('img', face_img)
        # cv2.waitKey(0)
        # print(np.array([face_img]).shape)
        pred = model.predict(np.array([face_img]))
        print(pred)
        return np.argmax(pred)
    return -1


dataset = '/home/prabodh/workspace/Face_Pose_Training/Prabodh/pose_data'
model_name = 'face_pose_v3_epochs_15_dropout'
# train_deep_neural_network(dataset)
train_cnn(dataset, model_name=model_name)

model = tf.keras.models.load_model(model_name)
classes = {0: 'frontal', 1: 'left', 2: 'right'}
img_path = '/home/prabodh/personal_space/How_I_Am_Learning_ML/Dataset/personal_images/my_face_1.JPG'
img_path2 = '/home/prabodh/personal_space/How_I_Am_Learning_ML/Dataset/personal_images/face_2.JPG'
img_path3 = '/home/prabodh/workspace/Face_Pose_Training/Shubham/face_pose_train_dataset/test/left/ND_labelled_v.vinothkumar@mphasis.com_mphasis_proctordesk_hosted_1415_.jpg'


def get_prediction(img_path):
    img = cv2.imread(img_path)
    ped = predict(model, image_numpy=img)
    # print(ped)
    img = cv2.putText(img, str(classes[ped]), (50, 50), cv2.FONT_HERSHEY_SIMPLEX,
                                               1, (255, 245, 0), 3, cv2.LINE_AA)
    cv2.imwrite(os.path.basename(img_path), img)
    return ped

[get_prediction(i) for i in [img_path, img_path2, img_path3]]

# How to increse accuracy
# 1. Hyperparameter tunning
# 2. Cross validation
# 3. Ensemble techniques