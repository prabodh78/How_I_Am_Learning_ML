import numpy as np
import cv2
import sys
import time
import os
import traceback
import glob


pd_prototxt_path = 'deep_models/person_det_ssd_arch.prototxt'
pd_model_path = 'deep_models/VGG_VOC0712_SSD_300x300_ft_iter_120000.caffemodel'

pd_net = cv2.dnn.readNetFromCaffe(pd_prototxt_path, pd_model_path)


def person_detector_ssd(net, image, detected_flag=False):
    (h, w) = image.shape[:2]
    # convert image format suitable for detector
    blob = cv2.dnn.blobFromImage(cv2.resize(image, (300, 300)), 1.0, (300, 300), (104, 117, 123))
    net.setInput(blob)
    detections = net.forward()
    # person class has the index 15
    person_class = detections[0][0][np.where(detections[0][0][:, 1] == 15)]
    # 0.84 is the threshold for the class "person"
    person_class_with_confidence = person_class[np.where(person_class[:, 2] > 0.84)].tolist()
    if person_class_with_confidence:
        # print ("person detected using SSD")
        detected_flag = True
        detection = person_class_with_confidence[0]
        box = detection[3:7] * np.array([w, h, w, h])
        startX, startY, endX, endY = box.astype("int")
        debug = False
        if debug is True:
            p1 = (int(startX), int(startY))
            p2 = (int(endX), int(endY))
            cv2.rectangle(image, p1, p2, (255, 0, 0), 2, 1)
            cv2.imshow('init', image)
            cv2.waitKey(0)
        return detected_flag, startX, startY, endX - startX, endY - startY, person_class_with_confidence
    else:
        return detected_flag, "None", "", "", "", person_class_with_confidence


img_path = '../../Dataset/personal_images/person_1.jpeg'
print(person_detector_ssd(pd_net, cv2.imread(img_path)))
