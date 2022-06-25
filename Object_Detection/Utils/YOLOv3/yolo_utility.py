import cv2
import numpy as np

# we are not going to bother with objects less than 30% probability
THRESHOLD = 0.3
# the lower the value: the fewer bounding boxes will remain
SUPPRESSION_THRESHOLD = 0.3
YOLO_IMAGE_SIZE = 320


def find_objects(model_outputs):
    bounding_box_locations = []
    class_ids = []
    confidence_values = []

    for output in model_outputs:
        for prediction in output:
            class_probabilities = prediction[5:]
            class_id = np.argmax(class_probabilities)
            confidence = class_probabilities[class_id]

            if confidence > THRESHOLD:
                w, h = int(prediction[2] * YOLO_IMAGE_SIZE), int(prediction[3] * YOLO_IMAGE_SIZE)
                # the center of the bounding box (we should transform these values)
                x, y = int(prediction[0] * YOLO_IMAGE_SIZE - w / 2), int(prediction[1] * YOLO_IMAGE_SIZE - h / 2)
                bounding_box_locations.append([x, y, w, h])
                class_ids.append(class_id)
                confidence_values.append(float(confidence))

    box_indexes_to_keep = cv2.dnn.NMSBoxes(bounding_box_locations, confidence_values, THRESHOLD, SUPPRESSION_THRESHOLD)

    return box_indexes_to_keep, bounding_box_locations, class_ids, confidence_values


def show_detected_images(img, bounding_box_ids, all_bounding_boxes, class_ids, confidence_values, width_ratio,
                         height_ratio):
    for index in bounding_box_ids:
        bounding_box = all_bounding_boxes[index[0]]
        x, y, w, h = int(bounding_box[0]), int(bounding_box[1]), int(bounding_box[2]), int(bounding_box[3])
        # we have to transform the locations adn coordinates because the resized image
        x = int(x*width_ratio)
        y = int(y * height_ratio)
        w = int(w * width_ratio)
        h = int(h * height_ratio)

        # OpenCV deals with BGR blue-green-red (255,0,0) then it is the blue color
        # we are not going to detect every objects just PERSON and CAR
        if class_ids[index[0]] == 2:
            cv2.rectangle(img, (x, y), (x+w, y+h), (255, 0, 0), 2)
            class_with_confidence = 'CAR' + str(int(confidence_values[index[0]] * 100)) + '%'
            cv2.putText(img, class_with_confidence, (x, y-10), cv2.FONT_HERSHEY_COMPLEX_SMALL, 0.5, (255, 0, 0), 1)

        if class_ids[index[0]] == 0:
            cv2.rectangle(img, (x, y), (x+w, y+h), (255, 0, 0), 2)
            class_with_confidence = 'PERSON' + str(int(confidence_values[index[0]] * 100)) + '%'
            cv2.putText(img, class_with_confidence, (x, y-10), cv2.FONT_HERSHEY_COMPLEX_SMALL, 0.5, (255, 0, 0), 1)


if __name__ == '__main__':
    image = cv2.imread('../Dataset/personal_images/person_1.jpeg')
    original_width, original_height = image.shape[1], image.shape[0]

    # there are 80 (90) possible output classes
    # 0: person - 2: car - 5: bus
    classes = ['car', 'person', 'bus']
    model_config, model_wights = 'yolov3.cfg', 'yolov3.weights'

    neural_network = cv2.dnn.readNetFromDarknet(model_config, model_wights)
    # define whether we run the algorithm with CPU or with GPU
    # WE ARE GOING TO USE CPU !!!
    neural_network.setPreferableBackend(cv2.dnn.DNN_BACKEND_OPENCV)
    neural_network.setPreferableTarget(cv2.dnn.DNN_TARGET_CPU)

    # the image into a BLOB [0-1] RGB - BGR
    blob = cv2.dnn.blobFromImage(image, 1 / 255, (YOLO_IMAGE_SIZE, YOLO_IMAGE_SIZE), True, crop=False)
    neural_network.setInput(blob)

    layer_names = neural_network.getLayerNames()
    # YOLO network has 3 output layer - note: these indexes are starting with 1
    output_names = [layer_names[index[0] - 1] for index in neural_network.getUnconnectedOutLayers()]

    outputs = neural_network.forward(output_names)
    predicted_objects, bbox_locations, class_label_ids, conf_values = find_objects(outputs)
    show_detected_images(image, predicted_objects, bbox_locations, class_label_ids, conf_values,
                         original_width / YOLO_IMAGE_SIZE, original_height / YOLO_IMAGE_SIZE)

    cv2.imshow('YOLO Algorithm', image)
    cv2.waitKey()
