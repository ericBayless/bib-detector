import cv2 as cv
import numpy as np

# Bib detection model config
bd_configPath = '../Data/YOLO/bib_detector/RBNR2_custom-yolov4-tiny-detector.cfg'
bd_weightsPath = '../Data/YOLO/bib_detector/RBNR2_custom-yolov4-tiny-detector_best.weights'
bd_classes = ['bib']

# Number reader config
nr_configPath = '../Data/YOLO/num_reader/SVHN3_custom-yolov4-tiny-detector.cfg'
nr_weightsPath = '../Data/YOLO/num_reader/SVHN3_custom-yolov4-tiny-detector_best.weights'
nr_classes = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]


class Detector:
    """
    Create YOLO object detection model in OpenCV with a given config and weights.
    Use this model to make predictions.
    """
    
    def __init__(self, cfg, wts, classes):
        
        self.classes = classes
        self.net = cv.dnn.readNetFromDarknet(cfg, wts)
        self.net.setPreferableBackend(cv.dnn.DNN_BACKEND_OPENCV)

        # determine the output layer
        self.ln = self.net.getLayerNames()
        self.ln = [self.ln[i[0] - 1] for i in self.net.getUnconnectedOutLayers()]
        
    def detect(self, img, conf):
        
        #format image for detection
        blob = cv.dnn.blobFromImage(img, 1/255.0, (416, 416), swapRB=True, crop=False)
        
         # get detections
        self.net.setInput(blob)
        outputs = self.net.forward(self.ln)

        # initialize lists
        boxes = []
        confidences = []
        classIDs = []

        # initialize image dimensions
        h_img, w_img = img.shape[:2]

        for output in outputs:
            for detection in output:
                scores = detection[5:]
                classID = np.argmax(scores)
                confidence = scores[classID]

                # drop low confidence detections and 
                if confidence > conf:
                    box = detection[:4] * np.array([w_img, h_img, w_img, h_img])
                    (centerX, centerY, width, height) = box.astype("int")
                    x = int(centerX - (width / 2))
                    y = int(centerY - (height / 2))
                    box = [x, y, int(width), int(height)]
                    boxes.append(box)
                    confidences.append(float(confidence))
                    classIDs.append(classID)

        # apply non maximal suppression for
        # initialize lists
        self.boxes = []
        self.confidences = []
        self.detected_classes = []
        cls_and_box = []
        # get indices of final bounding boxes  
        indices = cv.dnn.NMSBoxes(boxes, confidences, 0.5, 0.4)

        if len(indices) > 0:
            for i in indices.flatten():
                self.boxes.append(boxes[i])
                self.confidences.append(confidences[i])
                self.detected_classes.append(self.classes[classIDs[i]])
                
                cls_and_box.append([self.classes[classIDs[i]], boxes[i]])
        
        return cls_and_box


def get_bibs(img, single=False):

    # Instantiate detectors
    bd = Detector(bd_configPath, bd_weightsPath, bd_classes)
    nr = Detector(nr_configPath, nr_weightsPath, nr_classes)

    # Make bib location predictions
    bib_detections = bd.detect(img, 0.25)



    for obj in bib_detections:
        # crop out detected bib
        (x, y, w, h) = obj[1]
        obj.append(w * h)
        crop_img = img[y:y+h, x:x+w]
        
        # detect numbers on bib
        num_detections = nr.detect(crop_img, 0.5)
        bib_digit_loc = []
        if len(num_detections) > 0:
            # get digits and locations
            for digit in num_detections:
                (d_x, d_y, d_w, d_h) = digit[1]
                bib_digit_loc.append((d_x, str(digit[0])))

            # sort detected numbers L->R and put together
            bib_digit_loc.sort()
            rbn = int(''.join([i[1] for i in bib_digit_loc]))
            obj.append(rbn)
        else:
            obj.append(0)

    if single:
        bib_detections.sort(key=lambda x: x[2], reverse=True)
        return [bib_detections[0][3], bib_detections[0][1]]
    else:
        final_bibs = []
        for bib in bib_detections:
            final_bibs.append([bib[3], bib[1]])
        return final_bibs
