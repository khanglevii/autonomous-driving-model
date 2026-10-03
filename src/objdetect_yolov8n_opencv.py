import cv2
import os
import numpy as np


def detect(net, img, class_names='', imgsz=224, detect_thres=0.7):
    h, w = img.shape[:2]
    blob = cv2.dnn.blobFromImage(img, 1/255, (imgsz, imgsz), swapRB=True, crop=False)

    net.setInput(blob)
    out = net.forward()
    out = out[0] if isinstance(out, list) else out
    out = out.reshape(-1, out.shape[-1])  # flatten batch dim

    boxes = []
    confidences = []
    class_ids = []

    for det in out:
        obj_conf = float(det[4])
        if obj_conf < detect_thres:
            continue
        class_scores = det[5:]
        class_id = int(np.argmax(class_scores))
        class_conf = float(class_scores[class_id])
        conf = obj_conf * class_conf
        if conf < detect_thres:
            continue

        # Convert from YOLO format to pixel coordinates
        cx, cy, bw, bh = det[0:4]
        x1 = int((cx - bw / 2) * w)
        y1 = int((cy - bh / 2) * h)
        box_w = int(bw * w)
        box_h = int(bh * h)

        boxes.append([x1, y1, box_w, box_h])
        confidences.append(conf)
        class_ids.append(class_id)

    indices = cv2.dnn.NMSBoxes(boxes, confidences, detect_thres, 0.45)

    detections = []
    for i in indices:
        i = i[0] if isinstance(i, (tuple, list, np.ndarray)) else i
        detections.append((class_ids[i], confidences[i], boxes[i]))

    top4 = sorted(detections, key=lambda x: x[1], reverse=True)[:4]
    top4_classes = [class_names[i[0]] for i in top4]

    for cls_id, conf, box in top4:
        x, y, bw, bh = box
        label = f"{class_names[cls_id]}: {conf:.2f}"
        cv2.rectangle(img, (x, y), (x + bw, y + bh), (0, 255, 0), 2)
        cv2.putText(img, label, (x, y - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)

    return img, top4_classes


'''
def detect(net, img, class_names='', imgsz=224, detect_thres=0.7):
    top4_classes=0
    w = imgsz
    h = imgsz
    #Image blob
    blob = cv2.dnn.blobFromImage(img, 1/255, (w,h), swapRB=True, crop=False)

    # OpenCV DNN inference
    net.setInput(blob)
    out = net.forward() #Get detection data from the first output layer
    out = out[0] if isinstance(out, list) else out
    out = out.reshape(-1, out.shape[-1])
    print(out.shape)
    #Calculate confidence
    ''''''THEORY OF YOLOv8n OPERATION:
    STEP 1: Image blob is inputted into model
    STEP 2: Model returns a number of matrix layers based on how many object it detects from the input
    STEP 3: Each matrix has the structure: [x,y,object confidence, class0_confidence, ..., class(n-1)_confidence]
    With:
    x,y: Coordinate of center of the bounding box (Value: 0-1, float)
    w,h: Height & Width of the bounding box (Value: 0-1, float)
    object confidence: Probability of an object detected
    class_confidence: Probability of the class detected. Classes are ordered in the class_names file.
    STEP 4: DRAW BOUNDING BOX!!!
    ''''''
    
    boxes = []
    confidences = []
    class_ids = []
    for det in out:
        obj_conf = det[4] 
        if obj_conf < detect_thres:
            continue
        class_scores = det[5:]
        class_id = np.argmax(class_scores)
        class_conf = class_scores[class_id]
        conf = obj_conf * class_conf
        if conf < detect_thres:
            continue

        # YOLO x,y,w,h normalized format to pixels
        cx, cy, bw, bh = det[0:4] * np.array([w, h, w, h])
        x1, y1 = int(cx - bw / 2), int(cy - bh / 2)
        x2, y2 = int(cx + bw / 2), int(cy + bh / 2)
        boxes.append([x1,y1,x2-x1,y2-y1])
        confidences.append(float(conf))
        class_ids.append(class_id)
        
    indices = cv2.dnn.NMSBoxes(boxes, confidences, detect_thres, 0.45)
    
    detections = []
    for i in indices:
        box = boxes[i]
        cls_id = class_ids[i]
        conf = confidences[i]
        detections.append((cls_id, conf, box))
    
    top4 = sorted(detections, key=lambda x: x[1], reverse=True)[:4]
    top4_classes = [class_names[i[0]] for i in top4]
    for cls_id, conf, box in top4:
        x,y,w,h = box
        label = f"{class_names[cls_id]}: {conf:.2f}"
        cv2.rectangle(image, (x, y), (x + w, y + h), (0, 255, 0), 2)
        cv2.putText(image, label, (x, y - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
    return img, top4_classes
'''
