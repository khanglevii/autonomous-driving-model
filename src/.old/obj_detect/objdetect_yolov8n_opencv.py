import cv2
import os
import numpy as np
from ultralytics import YOLO

def detect(net, img, class_names='', imgsz=416, quantized_int=False, detect_thres=0.7):
    w = imgsz
    h = imgsz
    #Image blob
    if quantized_int == True: #For quantized integer models
        blob = cv2.dnn.blobFromImage(img, 1, (w,h), swapRB=True, crop=False) #1 is the scale of mapping the color range from [0:255] to [0:1] of each pixel
    else: #For FP32 or FP16 model
        blob = cv2.dnn.blobFromImage(img, 1/255, (w,h), swapRB=True, crop=False)

    # OpenCV DNN inference
    net.setInput(blob)
    out = net.forward()[0] #Get detection data from the first output layer

    #Calculate confidence
    '''
    THEORY OF YOLOv8n OPERATION:
    STEP 1: Image blob is inputted into model
    STEP 2: Model returns a number of matrix layers based on how many object it detects from the input
    STEP 3: Each matrix has the structure: [x,y,object confidence, class0_confidence, ..., class(n-1)_confidence]
    With:
    x,y: Coordinate of center of the bounding box (Value: 0-1, float)
    w,h: Height & Width of the bounding box (Value: 0-1, float)
    object confidence: Probability of an object detected
    class_confidence: Probability of the class detected. Classes are ordered in the class_names file.
    STEP 4: DRAW BOUNDING BOX!!!
    '''
    detect_class = []
    for det in out:
        conf = det[4] 
        if conf < detect_thres:
            continue
        class_scores = det[5:]
        class_id = np.argmax(class_scores)
        score = class_scores[class_id]
        if score < detect_thres:
            continue

        # YOLO x,y,w,h normalized format to pixels
        cx, cy, bw, bh = det[0:4] * np.array([w, h, w, h])
        x1, y1 = int(cx - bw / 2), int(cy - bh / 2)
        x2, y2 = int(cx + bw / 2), int(cy + bh / 2)
        
        label = f"{class_names[class_id]} {score:.2f}"
        cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(img, label, (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)

        detect_class.append(class_names[class_id]) #Add the detected class in the list
    
    top4_classes = []
    seen = set()
    for cls in sorted(detect_class, key=detect_class.count, reverse=True): #Get top 4 classes with most confidence rates
        if cls not in seen:
            top4_classes.append(cls)
            seen.add(cls)
        if len(top4_classes) == 4:
            break
    return img, top4_classes

