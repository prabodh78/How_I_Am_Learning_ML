from skimage import data, feature, transform
import matplotlib.pyplot as plt
from sklearn.svm import LinearSVC
from skimage.io import imread
from itertools import chain
from sklearn.datasets import fetch_lfw_people
from sklearn.feature_extraction.image import PatchExtractor
import numpy as np
from sklearn.metrics import accuracy_score, confusion_matrix

# we can load a data-set of human faces (positive samples)
human_faces = fetch_lfw_people()
positive_images = human_faces.images[:10000]
test_images = human_faces.images[10000:]
print(human_faces.images.shape, positive_images.shape, test_images.shape)

# fetch a data-set without faces (negative samples)

non_face_topics = ['moon', 'text', 'coins']

negative_samples = [(getattr(data, name)()) for name in non_face_topics]


# we will use PatchExtractor to generate several variants of these images
def generate_random_samples(image, num_of_generated_images=100, patch_size=positive_images[0].shape):
    extractor = PatchExtractor(patch_size=patch_size, max_patches=num_of_generated_images, random_state=42)
    patches = extractor.transform((image[np.newaxis]))
    return patches


def get_prediction(test_image):
    test_image_hog = np.array([feature.hog(test_image)])
    prediction = svm.predict(test_image_hog)
    # print("Prediction made by SVM: %f" % prediction)
    return int(prediction)


def model_evaluate():
    y_test = np.zeros(len(test_images))

    print('[Model-Evaluate] Y_test Shape: ', y_test.shape)
    for idx, img in enumerate(test_images):
        y_test[idx] = get_prediction(img)
    print('[Model-Evaluate] Accuracy : ', accuracy_score(np.ones(len(test_images)), y_test))
    cn_matrix = confusion_matrix(np.ones(len(test_images)), y_test)
    print('[Model-Evaluate] Confusion Matrix: \n', cn_matrix)
    print("[Model-Evaluate] TN, FP, FN, TP: ", cn_matrix.ravel())


# we generate 3000 samples (negative samples without a human face)
negative_images = np.vstack([generate_random_samples(im, 1000) for im in negative_samples])

# fig, axs = plt.subplots(10, 10)
# for i, axis in enumerate(axs.flat):
#     axis.imshow(negative_images[0], cmap='gray')
#     axis.axis('off')
# plt.show()

# we construct the training set with the output variables (labels)
# and of course we have to construct the HOG features
# TIME CONSUMING PROCEDURE !!!
X_train = np.array([feature.hog(image) for image in chain(positive_images, negative_images)])
# labels - 0 and 1 // 1: face 0: non-face
y_train = np.zeros(X_train.shape[0])
y_train[:positive_images.shape[0]] = 1

# print('Training Data Shape: ', X_train.shape)
# plt.imshow(X_train[0], cmap='gray')
# plt.show()
# we can construct the SVM
svm = LinearSVC()
# this is when SVM learns the parameters for the model based on the training dataset
svm.fit(X_train, y_train)

# read the test images
test_image = imread(fname='../Dataset/personal_images/person_1.jpeg')
test_image = transform.resize(test_image, positive_images[0].shape)

plt.imshow(test_image, cmap='gray')
plt.show()

test_image_hog = np.array([feature.hog(test_image)])
prediction = svm.predict(test_image_hog)
print("Prediction made by SVM: %f" % prediction)
model_evaluate()





