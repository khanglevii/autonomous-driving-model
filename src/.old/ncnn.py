import numpy as np
import cv2
from ultralytics import YOLO
from houghalgor import detect_lane_lines
from objdetect_yolov8n_ultralytics import *
from motortest import motorctrl
import os
import time
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--format',type=str,default='onnx',help=['onnx','ncnn','mnn'])
parser.add_argument('--precision',type=str,default='fp16',help=['fp32','fp16','int8'])
parser.add_argument('--modeltype',type=str,default='yolov8',help=['yolov3','yolov5','yolov8','yolov11'])
parser.add_argument('--modelversion',type=int,default=1)
args = parser.parse_args()

inference_mode = args.format
precision = args.precision
modeltype = args.modeltype
modelversion = args.modelversion

if inference_mode == 'ncnn': model_path = f"model/trafficspeedsign/ultralytics/{modeltype}/v{modelversion}/{inference_mode}/{precision}"
else: model_path = f"model/trafficspeedsign/ultralytics/{modeltype}/v{modelversion}/{inference_mode}/{precision}/trafficspeedsign_{precision}.{inference_mode}"
class_names_path = 'model/trafficspeedsign/obj.names' #Input class name file
detect_threshold = 0.7 #Detection threshold

net = YOLO(model=model_path,task='detect',verbose=False)

detect_imgsz = 224 #In WxH
show_imgsz = 320
cameraNo= 0

font = cv2.FONT_HERSHEY_PLAIN
cap = cv2.VideoCapture(cameraNo,cv2.CAP_V4L2)

frame_id = 0
ret = False
while True:
    frame_id += 1
    ret = False
    if frame_id % 4 == 0: #Read frame/4
        while ret == False: #Poll cam until success
            ret, frame = cap.read()
            if(ret == True): break
        xflip_frame = cv2.flip(frame, 0) #Horizontal flip
        yflip_frame = cv2.flip(xflip_frame, 1) #Vertical flip
        objimg = cv2.resize(yflip_frame, (detect_imgsz, detect_imgsz), cv2.INTER_NEAREST)
        t1 = time.perf_counter()
        results = net.predict(source=objimg,verbose=False,conf=detect_threshold)

        if detect_imgsz != show_imgsz:
            objimg = cv2.resize(objimg, (show_imgsz, show_imgsz), cv2.INTER_NEAREST)
            scale = show_imgsz / detect_imgsz
        else: scale = 1

        for result in results:
            boxes = result.boxes
            names = result.names

        for box in boxes:
            objconf = float(box.conf.cpu())
            if objconf < detect_threshold: continue
            objid = int(box.cls.cpu())
            x1,y1,x2,y2 = box.xyxy[0].cpu().numpy().astype(int)
            x1 = round(x1*scale)
            y1 = round(y1*scale)
            x2 = round(x2*scale)
            y2 = round(y2*scale)
            cv2.rectangle(objimg, (x1,y1), (x2,y2), color=(32,128,143), thickness=1)
            label = names.get(objid)
            cv2.putText(objimg,label,(x1,y1-10),cv2.FONT_HERSHEY_SIMPLEX,0.5,(0,255,0),2,cv2.LINE_AA)


        #print(names)
        #objimg, top4sign = detect(net,object_img, class_names, detect_imgsz, detect_threshold) #For using Ultralytics backend
        #objimg = cv2.resize(objimg, (show_imgsz,show_imgsz), cv2.INTER_LANCZOS4)
        t2 = time.perf_counter()
        rate = (t2 - t1)*1000 #Reaction time in millisec
        cv2.putText(objimg, f"Inference rate: {rate:.2f}ms", (10,30), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)

        # Display results
        cv2.imshow('Object Detection', objimg)

        if cv2.waitKey(1) & 0xFF == ord('q'): break

cap.release()
cv2.destroyAllWindows()
print('==> All done!')
print('***********************************************************')
