import cv2
import os
import numpy as np
from ultralytics import YOLO

def detect(net, img, class_names='', imgsz=416, detect_thres=0.7):
    # DNN inference
    results = net.predict(source=img, imgsz=imgsz, conf=detect_thres, verbose=False) #

    detect_class = []

    for result in results:
        boxes = result.boxes
        for box in boxes:
            cls_id = int(box.cls[0])
            conf = float(box.conf[0])
            if conf < detect_thres:
                continue

            # Bounding box coordinates
            x1, y1, x2, y2 = map(int, box.xyxy[0]) #Get bounding box coordinates
            label = f"{class_names[cls_id]} {conf:.2f}"
            cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(img, label, (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)

            detect_class.append(class_names[cls_id])

    # Get top 4 unique classes by frequency
    top4_classes = []
    seen = set()
    for cls in sorted(detect_class, key=detect_class.count, reverse=True):
        if cls not in seen:
            top4_classes.append(cls)
            seen.add(cls)
        if len(top4_classes) == 4:
            break

    return img, top4_classes
