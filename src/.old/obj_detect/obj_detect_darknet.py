import cv2
import os
import numpy as np

def detect(net, classes, font, output_layers, layers_names, colors, img):
    objclass = 0
    objconf = 0
    height, width, channels = img.shape
    blob = cv2.dnn.blobFromImage(img, 0.00392, (256, 256), (0, 0, 0), swapRB=True, crop=False)
    net.setInput(blob)
    outs = net.forward(output_layers)

    # Showing information on the screen
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
                cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0))

                boxes.append([x, y, w, h])
                confidences.append(float(confidence))
                # Name of the object
                class_ids.append(class_id)

    indexes = cv2.dnn.NMSBoxes(boxes, confidences, 0.5, 0.3)

    for i in range(len(boxes)):
        if i in indexes:
            x, y, w, h = boxes[i]
            label = "{}: {:.2f}%".format(classes[class_ids[i]], confidences[i] * 100)
            objclass = classes[class_ids[i]]
            objconf = confidences[i] * 100
            color = colors[i]
            cv2.rectangle(img, (x, y), (x + w, y + h), color, 2)
            cv2.putText(img, label, (x, y + 10), font, 1, color, 2)

    return img, objclass, objconf
