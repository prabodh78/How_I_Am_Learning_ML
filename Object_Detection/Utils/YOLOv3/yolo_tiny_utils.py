import cv2
import numpy as np
import random
import os
import matplotlib.pyplot as plt
import pandas as pd
import re
import time


wt = "yolo_tiny_improved_mobile_5_may_21/enet-coco-train_8000.weights"
cfg = "yolo_tiny_improved_mobile_5_may_21/enet-coco-train-higher-res.cfg"
net = cv2.dnn.readNet(wt, cfg)

classes = ['person', 'face', 'cellphone', 'hand']
layer_names = net.getLayerNames()
output_layers = [layer_names[i[0] - 1] for i in net.getUnconnectedOutLayers()]
colors = np.random.uniform(0, 255, size=(len(classes), 3))

font = cv2.FONT_HERSHEY_PLAIN


def yolo_prediction(img, pred_for='image'):
    orig_img = img
    if pred_for == 'image':
        orig_img = cv2.imread(img)
        img = cv2.imread(img)
    else:
        orig_img = img.copy()
    img = cv2.resize(img, None, fx=0.4, fy=0.4)
    height, width, channels = img.shape
    
    blob = cv2.dnn.blobFromImage(img, 0.00392, (416, 416), (0, 0, 0), True, crop=False)
    # blob = cv2.dnn.blobFromImage(img, size=(416, 416), swapRB=True, crop=False)
    net.setInput(blob)
    outs = net.forward(output_layers)
    
    class_ids = []
    confidences = []
    boxes = []
    for out in outs:
        for detection in out:
            scores = detection[5:]
            class_id = np.argmax(scores)
            confidence = scores[class_id]
            if confidence > 0.5:
                # Object detected
                center_x = int(detection[0] * width)
                center_y = int(detection[1] * height)
                w = int(detection[2] * width)
                h = int(detection[3] * height)
                # Rectangle coordinates
                x = int(center_x - w / 2)
                y = int(center_y - h / 2)
                boxes.append([x, y, w, h])
                confidences.append(float(confidence))
                class_ids.append(class_id)

    indexes = cv2.dnn.NMSBoxes(boxes, confidences, 0.5, 0.4)
    counter = 0
    final_boxes = []
    all_labels = []
    for i in range(len(boxes)):
        if i in indexes:
            x, y, w, h = boxes[i]
            label = str(classes[class_ids[i]])
            curr_class = class_ids[i]
            if curr_class == 1:
                color = (255, 0, 0)
            else:
                color = (0, 255, 0)
            color = colors[curr_class]
            x1 = int(x/0.4)
            y1 = int(y/0.4)
            x3 = int((x + w)/0.4)
            y3 = int((y + h)/0.4)
            cv2.rectangle(orig_img, (x1, y1), (x3, y3), color, 1)
            # print(confidences[i])
            cv2.putText(orig_img, label, (x1, y1), font, 0.5, color, 1)
            all_labels.append(label)
            final_boxes.append([curr_class, x/0.4, y/0.4, w/0.4, h/0.4])
    return final_boxes, all_labels


pred = yolo_prediction('../../Dataset/personal_images/person_1.jpeg')
print(pred)







