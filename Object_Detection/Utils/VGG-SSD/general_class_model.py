import cv2
import numpy as np
import glob, time, os


def filter_detections(posenet, image, detections, class_type, detector_threshold, nms_threshold, is_face=False):
    final_detections = []
    pose_index = None
    boxes = []
    confidences = []
    (h, w) = image.shape[:2]

    # person class has the index 1
    classes = detections[0][0][np.where(detections[0][0][:, 1] == class_type)]
    # 0.5 is the threshold for the class "person"

    class_with_confidence = classes[np.where(classes[:, 2] > detector_threshold[class_type])].tolist()
    # print("person_class_with_confidence: ", person_class_with_confidence)
    if class_with_confidence:
        # detections = person_class_with_confidence
        for detection in class_with_confidence:
            box = detection[3:7] * np.array([w, h, w, h])
            boxes.append(box)
            confidences.append(float(detection[2]))

    indices = cv2.dnn.NMSBoxes(boxes, confidences, detector_threshold[class_type], nms_threshold)
    # print person_class_with_confidence, '  ', indices
    try:
        count = indices.size
    except:
        count = 0
    if count > 0:
        if is_face:
            pose_index = get_face_pose(posenet, boxes, image)
        for index in indices:
            final_detections.append(class_with_confidence[index[0]])
        return final_detections, pose_index
    else:
        return [], None


def get_face_pose(posenet, rects, img):
    pose_index = None
    areas = [(rect[2] - rect[0]) * (rect[3] - rect[1]) for rect in rects]

    if areas:
        max_area_rect_idx = np.argmax(areas)
        x1, y1, x2, y2 = rects[max_area_rect_idx]
        croppedface = img[int(y1):int(y2), int(x1):int(x2)]
        # cv2.imshow('face', croppedface)
        try:
            posenet.setInput(cv2.dnn.blobFromImage(croppedface, 1.0, size=(32, 32), swapRB=True, crop=False))
            Poseout = posenet.forward()
            pose_index = np.argmax(Poseout)  # 2: Frontal, 0: Extreme Right, 1: Extreme Left
        except:
            pose_index = None

    return pose_index


def general_class_detector_nms(net, posenet, image, debug=False, detector_threshold=None, nms_threshold=0.3, name=None):

    if detector_threshold is None:
        detector_threshold = {0: 0.5, 1: 0.5, 2: 0.5, 3: 0.5}
    detector_result = {}

    image = cv2.resize(image, (640, 640))
    (h, w) = image.shape[:2]
    # convert image format suitable for detector
    blob = cv2.dnn.blobFromImage(image, size=(640, 640), swapRB=True, crop=False)
    # inputBlobImage = cv::dnn::blobFromImage(uMatImage_2, 1.0, cv::Size(300, 300), cv::Scalar(), true, false);
    # blob = cv2.dnn.blobFromImage(cv2.resize(image, (300, 300)), 1.0, (300, 300), (104, 117, 123))
    # blob = cv2.dnn.blobFromImage(cv2.resize(image, (300, 300)), 0.007843, (300, 300), (127.5, 127.5, 127.5), False)
    net.setInput(blob)
    detections = net.forward()

    for class_type in detector_threshold.keys():
        if class_type == 1:
            detector_result[class_type], pose_index = filter_detections(posenet, image, detections, class_type, detector_threshold,
                                                            nms_threshold, True)
        else:
            detector_result[class_type], _ = filter_detections(posenet, image, detections, class_type, detector_threshold, nms_threshold)

    if debug:
        image_copy = image.copy()
        classes = {0: 'Person', 1: 'Face', 2: 'Mobile', 3: 'Hand'}
        pose = {2: 'Frontal', 0: 'Ex Right', 1: 'Ex Left', 3: 'Not_sure'}
        for class_type, all_detection in detector_result.items():
            if all_detection:
                for detection in all_detection:
                    box = detection[3:7] * np.array([w, h, w, h])
                    startX, startY, endX, endY = box.astype("int")
                    p1 = (int(startX), int(startY))
                    p2 = (int(endX), int(endY))
                    cv2.rectangle(image, p1, p2, (36, 255, 12), 2, 1)
                    if class_type == 1:
                        text = str(classes[class_type])+': ' + str(round(detection[2], 2)) + ' Pose: ' + str(pose[pose_index])
                    else:
                        text = str(classes[class_type]) + ': ' + str(round(detection[2], 2))

                    cv2.putText(image, text, (startX, startY - 5), cv2.FONT_HERSHEY_SIMPLEX,
                                0.5, (255, 0, 0), 2)

        cv2.imshow('Img', image); cv2.waitKey(0);cv2.destroyAllWindows()
        # iou = 0
        # if detector_result[1] and detector_result[3]:
        #     for face_bbx in detector_result[1]:
        #         face_bbx = face_bbx[3:7] * np.array([w, h, w, h])
        #         face_bbx = face_bbx.astype("int")
        #         for hand_bbx in detector_result[3]:
        #             hand_bbx = hand_bbx[3:7] * np.array([w, h, w, h])
        #             hand_bbx = hand_bbx.astype("int")
        #             iou2 = bb_intersection_over_union(face_bbx, hand_bbx)
        #             if iou2 > iou:
        #                 iou = iou2
        # print("IOU: ", iou)
        # cv2.imshow('Opencv.jpg', image)

    return detector_result, pose_index


def bb_intersection_over_union(boxA, boxB):
    # determine the (x, y)-coordinates of the intersection rectangle
    xA = max(boxA[0], boxB[0])
    yA = max(boxA[1], boxB[1])
    xB = min(boxA[2], boxB[2])
    yB = min(boxA[3], boxB[3])
    # compute the area of intersection rectangle
    interArea = max(0, xB - xA + 1) * max(0, yB - yA + 1)
    # compute the area of both the prediction and ground-truth
    # rectangles
    boxAArea = (boxA[2] - boxA[0] + 1) * (boxA[3] - boxA[1] + 1)
    boxBArea = (boxB[2] - boxB[0] + 1) * (boxB[3] - boxB[1] + 1)
    # compute the intersection over union by taking the intersection
    # area and dividing it by the sum of prediction + ground-truth
    # areas - the interesection area
    iou = interArea / float(boxAArea + boxBArea - interArea)
    # return the intersection over union value
    return iou


if __name__ == '__main__':
    pd_prototxt_path = 'deep_models/infer_faster_rcnn_resnet50_49838_31MAP.pbtxt'
    pd_model_path = 'deep_models/infer_faster_rcnn_resnet50_49838_31MAP.pb'
    # Loading general class model
    model = cv2.dnn.readNetFromTensorflow(pd_model_path, pd_prototxt_path)
    # Loading pose classifier model
    posenet = cv2.dnn.readNetFromTensorflow(
                                    'deep_models/Best-weights-resent10_with_reg_for_pic_gen_class-001-0.1168-0.9991.pb')

    detector_threshold = {0: 0.5, 1: 0.5, 2: 0.5, 3: 0.5}
    images_path = ['../../Dataset/personal_images/person_1.jpeg']
    for image_path in images_path:
        img = cv2.imread(image_path)
        name = os.path.basename(image_path)
        t1 = time.time()
        # 0: Person 1: Face
        result, pose_index = general_class_detector_nms(model, posenet, img, True,
                                                        detector_threshold=detector_threshold, nms_threshold=0.30)
