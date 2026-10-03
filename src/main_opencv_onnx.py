import numpy as np
import cv2
from houghalgor import detect_lane_lines
from objdetect_yolov8n_opencv import *
from motortest import motorctrl
import os

model_path = 'model/Traffic+Speed Sign/v9(yolov8)/opencv_onnx/fp16/trafficspeedsign_fp16.onnx' #Input model path (.onnx)
class_names_path = 'model/Traffic+Speed Sign/v3/obj.names' #Input class name file
with open(class_names_path, 'r') as f:
    class_names = [line.strip() for line in f]
detect_threshold = 0.7 #Detection threshold

frameWidth= 640
frameHeight = 480
detect_imgsz = 224
cameraNo= 0
original_speed = 50
speed_100 = 0.8*original_speed
speed_80 = 0.6*original_speed
speed_60 = 0.4*original_speed
speed_yellow = 0.2*original_speed
stopspeed = 0
speed =  original_speed#Init speed
maxspeed = 100 #Max speed of the car
motor = motorctrl(In1A=22, In1B=23, In2A=24, In2B=25)

font = cv2.FONT_HERSHEY_PLAIN
cap = cv2.VideoCapture(cameraNo,cv2.CAP_V4L2)

net = cv2.dnn.readNetFromONNX(model_path) #Load model first (OPENCV)
net.setPreferableBackend(cv2.dnn.DNN_BACKEND_OPENCV)
net.setPreferableTarget(cv2.dnn.DNN_TARGET_CPU_FP16)

frame_id = 0

while True:
    frame_id += 1
    ret = False
    if frame_id % 4 == 0: #Read frame/4
        while ret == False: #Poll cam until success
            ret, frame = cap.read()
            if(ret == True): break
        xflip_frame = cv2.flip(frame, 0) #Horizontal flip
        yflip_frame = cv2.flip(xflip_frame, 1) #Vertical flip
        img = cv2.resize(yflip_frame, (frameWidth,frameHeight))
        object_img = cv2.resize(yflip_frame, (detect_imgsz,detect_imgsz))
        
        steering_angle, lanes_overlay = detect_lane_lines(img)
        motor.move(speed,maxspeed,0.01*steering_angle)
        sign = 0
        objimg, top4sign = detect(net, object_img, class_names, detect_imgsz, detect_threshold) #For using OPENCV backend
        if ((sign == "Stop" or sign == "Red  Light") and confidence > 40): speed = stopspeed
        if (sign == "speed_limit_100" and confidence > 60): speed = speed_100
        if (sign == "speed_limit_80" and confidence > 60): speed = speed_80
        if (sign == "speed_limit_60" and confidence > 60): speed = speed_60
        if (sign == "yellow Light" and confidence > 50): speed = speed_yellow
        if (sign == "Green Light" and confidence > 60): speed = original_speed
        
        
        # Display results
        cv2.putText(lanes_overlay, f"Steering Angle: {steering_angle:.2f}", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
        cv2.imshow('Lane Detection', lanes_overlay)
        #cv2.imshow('Object Detection', objimg)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
motor.stop()
print('==> All done!')
print('***********************************************************')
