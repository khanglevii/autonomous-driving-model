import numpy as np
import cv2
from ultralytics import YOLO
from houghalgor import detect_lane_lines
import objdetect_ultralytics as cvs
from motortest import motorctrl
import os
import time
import argparse
import gc
import torch

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

if inference_mode == 'ncnn': model_path = f"./model/trafficspeedsign/ultralytics/{modeltype}/v{modelversion}/{inference_mode}/{precision}_ncnn_model"
else: model_path = f"model/trafficspeedsign/ultralytics/{modeltype}/v{modelversion}/{inference_mode}/{precision}/trafficspeedsign_{precision}.{inference_mode}"

detect_threshold = 0.6 #Detection threshold
dist_threshold = 1
camL,camR = 1,0 #CamR is for YOLO&Lane
stereo_calib_file = 'assets/calib/stereo_maps.npz'
line_calib_file = 'assets/calib/line_maps.npz'
lane_imgsz = 160, 120
detect_imgsz = 128,128
show_imgsz = 640,480

#Speed level
original_speed = 90
maxspeed = 200 #Max speed of the car
speedlvl = [0,70,80,original_speed] #Stop, Slow, Medium, Max
speed = speedlvl[3] #Initial speed (Max)

motor = motorctrl(In1A=22, In1B=23, In2A=25, In2B=24)

vision = cvs.vision(original_speed,camL,camR,stereo_calib_file,line_calib_file,model_path,detect_threshold,dist_threshold,lane_imgsz,detect_imgsz,show_imgsz)
cpu_thread = os.cpu_count()
print(f"CPU threads: {cpu_thread}")
cv2.setNumThreads(3)
cv2.setUseOptimized(True)
print(cv2.useOptimized())
frame_id = 0
stop = True
previouserror=0

while True:
    frame_id += 1
    vision.capture() #Take pictures from cameras.
    detect_spd_lvl = vision.detect() #Speed change according to traffic light/sign.
    
    estop, dodge_angle = vision.depth_est() #Determine dodge angle
    
    
    if stop == True or estop == True: #Manual stop / Emergency stop
        cvs.speed = 0 #To show speed on GUI
        vision.lanectrl(-1)
        motor.move(0,0,0) #Motor brake
    
    else: #Operating state
        match detect_spd_lvl:
            case 0: #Stop / Red light
                speed = speedlvl[0]
            case 1: #60km/h / Yellow light
                speed = speedlvl[1]
            case 2: #80km/h
                speed = speedlvl[2]
            case 3: #100km/h / Green light
                speed = speedlvl[3]
                
        cvs.speed = speed #To show speed on GUI        
        if dodge_angle != 0: motor.move(speed,maxspeed, -dodge_angle*40)
        else:
            steering_angles = []
            for i in range(2):
                steering_angle = vision.lanectrl(5)
                steering_angles.append(steering_angle)
            pidval = np.mean(steering_angles)
            
            '''Kp = 8
            Kd = 0.4
            Ki = 0.0001
            
            p=error;
            i=i+error;
            d=error-previouserror;
            previouserror=error;
            pidval=p*Kp+i*Ki+d*Kd;
            if pidval > maxspeed: pidval = maxspeed
            if pidval < -maxspeed : pidval = -maxspeed''' 
            motor.move(speed,maxspeed, pidval)
    
    vision.maingui()
    
    # Display results
    if frame_id % 1024 == 0:
            frame_id = 0
            gc.collect()
            torch.cuda.empty_cache()
            
    key = cv2.waitKey(1)
    match key:
            case 27: break
            case 32 : stop = ~stop

vision.close()
cv2.destroyAllWindows()
motor.stop()
