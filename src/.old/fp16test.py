import numpy as np
import cv2
from houghalgor import detect_lane_lines
from obj_detect import *
#from motortest import Motor
import os
import time
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--model_cfg', type = str, default = 'model/darknetyolo/speedsign/speedsign.cfg',
                    help = 'Path to config file')
parser.add_argument('--model_weights', type=str,
                    default='model/darknetyolo/speedsign/speedsign.weights',
                    help='Path to weight of model')
args = parser.parse_args()

# print the arguments
print('----- info -----')
print('[i] The config file: ', args.model_cfg)
print('[i] The weights of model file: ', args.model_weights)
print('###########################################################\n')
class_txt_files = "model/darknetyolo/speedsign/speedsign.names"
frameWidth= 640
frameHeight = 480
detectWidth = 512
detectHeight = 512 
cameraNo= 0
original_speed = 50
speed_100 = 0.8*original_speed
speed_80 = 0.6*original_speed
speed_60 = 0.4*original_speed
speed_yellow = 0.2*original_speed
stopspeed = 0;
speed =  original_speed#Init speed
maxspeed = 100 #Max speed of the car
#motor = Motor(In1A=27, In2A=22, In1B=3, In2B=4)

font = cv2.FONT_HERSHEY_PLAIN
cap = cv2.VideoCapture(cameraNo,cv2.CAP_V4L2)

net = cv2.dnn.readNet(args.model_weights, args.model_cfg)
net.setPreferableBackend(cv2.dnn.DNN_BACKEND_OPENCV)
#net.setPreferableTarget(cv2.dnn.DNN_TARGET_CPU_FP16)

classes = []
with open(class_txt_files, "r") as f:
    classes = [line.strip() for line in f.readlines()] # we put the names in to an array
layers_names = net.getLayerNames()
output_layers = net.getUnconnectedOutLayersNames()
colors = np.random.uniform(0, 255, size = (len(classes), 3))

frame_id = 0


ret = False
while ret == False: #Poll cam until success
	ret, frame = cap.read()
	if(ret == True): break
	
xflip_frame = cv2.flip(frame, 0) #Horizontal flip
yflip_frame = cv2.flip(xflip_frame, 1) #Vertical flip
img = cv2.resize(yflip_frame, (frameWidth,frameHeight))

blob = cv2.dnn.blobFromImage(img, 0.00392, (256, 256), (0, 0, 0), swapRB=True, crop=False)

net.setPreferableTarget(cv2.dnn.DNN_TARGET_CPU)
net.setInput(blob)
start = time.time()
out = net.forward()
print("CPU:", time.time() - start)

# Try FP16
net.setPreferableTarget(cv2.dnn.DNN_TARGET_CPU_FP16)
net.setInput(blob)
start = time.time()
out = net.forward()
print("CPU FP16:", time.time() - start)

# Try OPENCL
net.setPreferableTarget(cv2.dnn.DNN_TARGET_OPENCL)
net.setInput(blob)
start = time.time()
out = net.forward()
print("OPENCL:", time.time() - start)

# Try OPENCL FP16
net.setPreferableTarget(cv2.dnn.DNN_TARGET_OPENCL_FP16)
net.setInput(blob)
start = time.time()
out = net.forward()
print("OPENCL FP16:", time.time() - start)

# Try VULKAN
net.setPreferableBackend(cv2.dnn.DNN_BACKEND_VKCOM)
net.setPreferableTarget(cv2.dnn.DNN_TARGET_VULKAN)
net.setInput(blob)
start = time.time()
out = net.forward()
print("VULKAN:", time.time() - start)

'''while True:
    frame_id += 1
    ret = False
    if frame_id % 4 == 0: #Read frame/4
        while ret == False: #Poll cam until success
            ret, frame = cap.read()
            if(ret == True): break
        xflip_frame = cv2.flip(frame, 0) #Horizontal flip
        yflip_frame = cv2.flip(xflip_frame, 1) #Vertical flip
        img = cv2.resize(yflip_frame, (frameWidth,frameHeight))
        object_img = cv2.resize(yflip_frame, (detectWidth,detectHeight))
        
        steering_angle, lanes_overlay = detect_lane_lines(img)
        #motor.move(speed,maxspeed,0.01*steering_angle)
        sign = 0
        objimg, sign, confidence = detect(net,classes,font,output_layers,layers_names,colors,object_img)
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
        cv2.imshow('Object Detection', objimg)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
#motor.stop()
print('==> All done!')
print('***********************************************************')
'''


