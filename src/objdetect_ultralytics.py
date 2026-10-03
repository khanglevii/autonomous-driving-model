import cv2
import time
import numpy as np
import math
import statistics as st
from picamera2 import Picamera2
from ultralytics import YOLO

'''
Rules of resizing images:
1. For display: Use cv2.INTER_NEAREST (FASTEST, SHITTY)
2. For downscale and input: Use cv2.INTER_AREA (SLOW, ACCURATE)
3. For upscale and input: Use cv2.INTER_LANCZOS4 (SLOW, ACCURATE)
4. For changing resolution (i.e 640x480 -> 480x640): Use cv2.INTER_LINEAR (FAST, MEDIUM)
5. For changing resolution (i.e 640x480 -> 480x640): Use cv2.INTER_CUBIC (MEDIUM--, GOOD)

Rules of remapping images:
1. For stereo input: Use cv2.INTER_LINEAR (FAST, MEDIUM)
'''
speed = 0
class vision:
    def __init__(self,orgspeed,camL,camR,stereo_calib_file,line_calib_file,model_path,minconf,mindist,lane_imgsz,detect_imgsz,gui_imgsz=None):
         #Imgsz standard: H,W (Numpy standard)
        self.lane_imgsz = lane_imgsz
        self.detect_imgsz = detect_imgsz[::-1]
        if gui_imgsz != None: #Set default output size
            self.gui_imgsz = gui_imgsz
        else: self.gui_imgsz = 640,480
        
        self.minconf = minconf
        self.font = cv2.FONT_HERSHEY_SIMPLEX
        self.mindist = mindist
        self.orgspeed = orgspeed
        #Load model first (Ultralytics)
        self.net = YOLO(model=model_path,task='detect')
        
        #Setup camera
        self.camL = Picamera2(camL)
        self.camR = Picamera2(camR)
        configL = self.camL.create_preview_configuration(main={"size": (640,480)})
        configR = self.camR.create_preview_configuration(main={"size": (640,480)})
        self.camL.configure(configL)
        self.camR.configure(configR)
        self.camL.start()
        self.camR.start()
        
        #Load stereo calibration data
        stereo_cal = np.load(stereo_calib_file)
        self.stereomapLx, self.stereomapLy, self.stereomapRx, self.stereomapRy, self.stereoQ, self.stereocalres = [stereo_cal[f] for f in ("mapLx","mapLy","mapRx","mapRy","Q","calres")]
        line_cal = np.load(line_calib_file)
        self.linemapLx, self.linemapLy, self.linemapRx, self.linemapRy, self.lineQ, self.linecalres = [line_cal[f] for f in ("mapLx","mapLy","mapRx","mapRy","Q","calres")]
        
        min_disp = 0
        num_disp = 32 #must be multiple of 16
        block_size = 5# must be odd, usually 5-21 works best
        
        self.stereo = cv2.StereoSGBM_create(
            minDisparity=min_disp,
            numDisparities=num_disp,
            blockSize=block_size,
            P1=8 * block_size**2,
            P2=32 * block_size**2,
            disp12MaxDiff=1,
            uniquenessRatio=10,
            speckleWindowSize=50,
            speckleRange=2,
            preFilterCap=31,
            mode=cv2.STEREO_SGBM_MODE_SGBM_3WAY
        )
        
        self.lanegui = np.zeros((self.gui_imgsz[1], self.gui_imgsz[0], 4), dtype=np.uint8) #BGRA
        self.objdetectgui = np.zeros((self.gui_imgsz[1], self.gui_imgsz[0], 4), dtype=np.uint8) #BGRA
        
        self.linesrc = 0 #0: Rectified R, 1: Raw R
        
    def capture(self):
        self.frameL = self.camL.capture_array() #CamR for lane+object detection
        self.frameR = self.camR.capture_array() #CamL+R for distance estimation
        self.frameL = cv2.cvtColor(self.frameL, cv2.COLOR_RGB2BGR)
        self.frameR = cv2.cvtColor(self.frameR, cv2.COLOR_RGB2BGR)
        
        #Color correction
        green_scale = 0.91
        self.frameL[:, :, 1] = np.clip(self.frameL[:, :, 1] * green_scale, 0, 255).astype(np.uint8)
        green_scale = 0.82
        self.frameR[:, :, 1] = np.clip(self.frameR[:, :, 1] * green_scale, 0, 255).astype(np.uint8)
        
        if self.linesrc == 0: #For using remapped src
            self.remappedL = cv2.resize(self.frameL,(self.linecalres),cv2.INTER_LINEAR)
            self.remappedL = cv2.remap(self.remappedL,self.linemapLx,self.linemapLy,cv2.INTER_LINEAR)
            self.remappedR = cv2.resize(self.frameR,(self.linecalres),cv2.INTER_LINEAR) #If different from mapping calib size
            self.remappedR = cv2.remap(self.remappedR,self.linemapRx,self.linemapRy,cv2.INTER_LINEAR)
            
        match self.linesrc:
            case 0: #Rectified R
                self.gui = cv2.cvtColor(self.remappedR, cv2.COLOR_BGR2BGRA)
            case 1: #Raw R
                self.gui = cv2.cvtColor(self.frameR, cv2.COLOR_BGR2BGRA)
                
        if  self.gui.shape[:2][::-1] != self.gui_imgsz: self.gui = cv2.resize(self.gui, self.gui_imgsz, cv2.INTER_NEAREST)
        
    def maingui(self):
        debug = True
        result = self.gui.astype(np.float32)
        overlays = [self.objdetectgui, self.lanegui]

        for overlay in overlays:
            alpha = overlay[:, :, 3:] / 255.0  # normalize to 0-1
            inv_alpha = 1.0 - alpha

            # Broadcast alpha to 3 channels for color blending
            alpha_3 = np.repeat(alpha, 3, axis=2)
            inv_alpha_3 = np.repeat(inv_alpha, 3, axis=2)

            # Blend RGB channels
            result[:, :, :3] = (
                inv_alpha_3 * result[:, :, :3] +
                alpha_3 * overlay[:, :, :3]
            )
        
        
        result = np.clip(result, 0, 255).astype(np.uint8)
        result = np.hstack((self.guidisp,result))
        
        #Create info bar on the far right
        h,w = result.shape[:2]
        infobar = np.zeros((h,w//8,4), dtype=np.uint8) #BGRA
        result = np.hstack((result,infobar))
        
        #Add info to the infobar
        h,w = result.shape[:2]
        
        #Basic info
        offset = 20
        cv2.putText(result,f"Angle: {self.steering_angle}",(int(0.89*w),offset),self.font,0.5,(0,255,0,255),1)
        cv2.putText(result,f"Speed: {speed}/255",(int(0.89*w),14+offset),self.font,0.5,(0,255,0,255),1)
        
        
        
        if debug == True:
            offset = 55
            cv2.putText(result,"Rate:",(int(0.89*w),offset),self.font,0.5,(127,0,255,255),1)
            cv2.putText(result,f"DIST: {self.distreactrate:.2f}ms",(int(0.9*w),13+offset),self.font,0.5,(127,0,255,255),1)
            cv2.putText(result,f"DET: {self.detreactrate:.2f}ms",(int(0.9*w),30+offset),self.font,0.5,(127,0,255,255),1)
            cv2.putText(result,f"LINE: {self.lanereactrate:.2f}ms",(int(0.9*w),45+offset),self.font,0.5,(127,0,255,255),1)
            #Rescale mode
            offset = 127
            cv2.putText(result,"RESMODE:",(int(0.89*w),offset),self.font,0.5,(0,0,255,255),1)
            cv2.putText(result,f"DIST: {self.stereoresmode}",(int(0.9*w),13+offset),self.font,0.5,(0,0,255,255),1)
            cv2.putText(result,f"DET: {self.detectresmode}",(int(0.9*w),30+offset),self.font,0.5,(0,0,255,255),1)
            cv2.putText(result,f"LINE: {self.detectresmode}",(int(0.9*w),45+offset),self.font,0.5,(0,0,255,255),1)
            
            #Operating resolution
            offset = 200
            cv2.putText(result,"OPRES:",(int(0.89*w),offset),self.font,0.5,(0,127,255,255),1)
            cv2.putText(result,f"DIST: {self.stereocalres[0]}x{self.stereocalres[1]}",(int(0.9*w),17+offset),self.font,0.5,(0,127,255,255),1)
            cv2.putText(result,f"DET: {self.detect_imgsz[0]}x{self.detect_imgsz[1]}",(int(0.9*w),32+offset),self.font,0.5,(0,127,255,255),1)
            cv2.putText(result,f"LINE: {self.lane_imgsz[0]}x{self.lane_imgsz[1]}",(int(0.9*w),47+offset),self.font,0.5,(0,127,255,255),1)
            cv2.putText(result,f"GUI: {self.gui_imgsz[0]}x{self.gui_imgsz[1]}",(int(0.9*w),60+offset),self.font,0.5,(0,127,255,255),1)
        
        cv2.imshow("Main GUI", result)
        
    def depth_est(self): #Depth estimation
        debug = True
        t1 = time.perf_counter()
        #Input image scaling
        h,w = self.frameL.shape[:2] #Extract size
        if w>self.stereocalres[0] and h>self.stereocalres[1]: #Downscaling for faster speed
            self.stereoresmode = 'AREA'
            rectL = cv2.resize(self.frameL,self.stereocalres,interpolation=cv2.INTER_AREA)
            rectR = cv2.resize(self.frameR,self.stereocalres,interpolation=cv2.INTER_AREA)
        elif w<self.stereocalres[0] and h<self.stereocalres[1]: #Shitty pics need upscaling
            self.stereoresmode = 'LANCZOS'
            rectL = cv2.resize(self.frameL,self.stereocalres,interpolation=cv2.INTER_LANCZOS4)
            rectR = cv2.resize(self.frameR,self.stereocalres,interpolation=cv2.INTER_LANCZOS4)
        else:
            self.stereoresmode = 'CUBIC'
            rectL = cv2.resize(self.frameL,self.stereocalres,interpolation=cv2.INTER_CUBIC)
            rectR = cv2.resize(self.frameR,self.stereocalres,interpolation=cv2.INTER_CUBIC)
        
        #Stereo cameras calibration
        rectL = cv2.remap(rectL, self.stereomapLx, self.stereomapLy, cv2.INTER_CUBIC)
        rectR = cv2.remap(rectR, self.stereomapRx, self.stereomapRy, cv2.INTER_CUBIC)
        
        # Grayscale for disparity
        grayL = cv2.cvtColor(rectL, cv2.COLOR_BGR2GRAY)
        grayR = cv2.cvtColor(rectR, cv2.COLOR_BGR2GRAY)
        
        #Preprocessing steps
        grayL = cv2.equalizeHist(grayL)
        grayR = cv2.equalizeHist(grayR)
        
        blur=7,7
        grayL = cv2.GaussianBlur(grayL, blur, 2)
        grayR = cv2.GaussianBlur(grayR, blur, 2)
        #cv2.imshow('gauss',np.hstack((grayL,grayR)))
        
        grayL = cv2.equalizeHist(grayL)
        grayR = cv2.equalizeHist(grayR)
        
        # Compute disparity
        disparity = self.stereo.compute(grayL, grayR).astype(np.float32) / 16.0 #Divide 16 since OPENCV store disparity as int16, by calculate in x16.
        
        #Postprocessing steps
        disparity = cv2.medianBlur(disparity, 5) #Apply median blur
        
        #disparity = cv2.bilateralFilter(disparity, 9, 75, 75)
        kernel = np.ones((5,5),np.uint8)
        disparity = cv2.morphologyEx(disparity, cv2.MORPH_CLOSE, kernel)
        #cv2.imshow('disparity',disparity)
        # Reproject to 3D
        points_3D = cv2.reprojectImageTo3D(disparity, self.stereoQ)
        
        h,w = disparity.shape
        
        # ROI instead of single pixel (for stability)
        roi_size = 7
        #Chia thanh 3 mien
        lx, ly = (w//4), h // 2 #Left region
        cx, cy = w // 2, h // 2 # Center region
        rx, ry = (3*w//4), h // 2 #Right region
        offsetx = 0
        offsety = -30
        
        roiL = points_3D[ly - roi_size + offsety:ly + roi_size + offsety, lx - roi_size:lx + roi_size, :]
        roiC = points_3D[cy - roi_size + offsety:cy + roi_size + offsety, cx - roi_size:cx + roi_size, :]
        roiR = points_3D[ry - roi_size + offsety:ry + roi_size + offsety, rx - roi_size:rx + roi_size, :]
        
        # Extract Z (depth)
        ZL = roiL[:, :, 2]
        ZC = roiC[:, :, 2]
        ZR = roiR[:, :, 2]
        
        # remove invalid values (inf/nan)
        ZL = ZL[np.isfinite(ZL)]
        ZC = ZC[np.isfinite(ZC)]
        ZR = ZR[np.isfinite(ZR)]
        
        ZL = float(np.median(ZL))
        ZC = float(np.median(ZC))
        ZR = float(np.median(ZR))
        
        if ZL < 0: ZL = np.nan
        if ZC < 0: ZC = np.nan
        if ZR < 0: ZR = np.nan
        
        if ZL != np.nan: ZL -= 0.1
        if ZC != np.nan: ZC -= 0.1
        if ZR != np.nan: ZR -= 0.1
        
        # Draw crosshair
        guidisp = rectL
        scale_w = self.gui_imgsz[0] / self.stereocalres[0]
        scale_h = self.gui_imgsz[1] / self.stereocalres[1]
        
        # Normalize disparity for display
        if debug == False:
            disp_vis = cv2.normalize(disparity, None, 0, 255, cv2.NORM_MINMAX)
            disp_vis = np.uint8(disp_vis)
            disp_vis = cv2.resize(disp_vis,self.gui_imgsz,interpolation=cv2.INTER_NEAREST) #Upscale disparity map for display
            guidisp = cv2.resize(guidisp,self.gui_imgsz,interpolation=cv2.INTER_NEAREST) #Upscale rectified left image for display
            guidisp = np.hstack((guidisp, cv2.cvtColor(disp_vis, cv2.COLOR_GRAY2BGR)))
        else:
            guidisp = cv2.resize(guidisp,self.gui_imgsz,interpolation=cv2.INTER_NEAREST) #Upscale rectified left image for display
        
        cv2.rectangle(guidisp, (int((lx - roi_size)*scale_w),int((ly - roi_size + offsety)*scale_h)),(int((lx + roi_size)*scale_w),int((ly + roi_size + offsety)*scale_h)),(0,0,255,0),thickness=1)
        cv2.rectangle(guidisp, (int((cx - roi_size)*scale_w),int((cy - roi_size + offsety)*scale_h)),(int((cx + roi_size)*scale_w),int((cy + roi_size + offsety)*scale_h)),(0,0,255,0),thickness=1)
        cv2.rectangle(guidisp, (int((rx - roi_size)*scale_w),int((ry - roi_size + offsety)*scale_h)),(int((rx + roi_size)*scale_w),int((ry + roi_size + offsety)*scale_h)),(0,0,255,0),thickness=1)
        
        cv2.putText(guidisp, f"{ZL:.2f} m", (int((lx - roi_size - 5)*scale_w),int((ly - roi_size - 5)*scale_h)), self.font, 1, (0,255,0), 2, cv2.LINE_AA)
        cv2.putText(guidisp, f"{ZC:.2f} m", (int((cx - roi_size - 5)*scale_w),int((cy - roi_size - 5)*scale_h)), self.font, 1, (0,255,0), 2, cv2.LINE_AA)
        cv2.putText(guidisp, f"{ZR:.2f} m", (int((rx - roi_size - 5)*scale_w),int((ry - roi_size - 5)*scale_h)), self.font, 1, (0,255,0), 2, cv2.LINE_AA)
        
        
        if ZL<self.mindist and ZC>self.mindist and ZR>self.mindist: #Dodge little right
            stop = False
            steering_angle = 2
        elif ZL<self.mindist and ZC<self.mindist and ZR>self.mindist: #Dodge right
            stop = False
            steering_angle = 4
        elif ZL>self.mindist and ZC>self.mindist and ZR<self.mindist: #Dodge little left
            stop = False
            steering_angle = -2
        elif ZL>self.mindist and ZC<self.mindist and ZR<self.mindist: #Dodge left
            stop = False
            steering_angle = -4
        elif ZL<self.mindist and ZC<self.mindist and ZR<self.mindist: #Stuck
            stop = True
            steering_angle = 0
        elif ZL>self.mindist and ZC<self.mindist and ZR>self.mindist: #Dodge left (Dodge center)
            stop = False
            steering_angle = -2
        else:
            stop = False
            steering_angle = 0
            
        #cv2.putText(guidisp, f"CalRes: {self.stereocalres}", (10,80), self.font, 0.5, (0,255,127), 2, cv2.LINE_AA)
        t2 = time.perf_counter()
        self.distreactrate = (t2 - t1)*1000 #Reaction time in millisec
        #cv2.putText(guidisp, f"Rate: {rate:.2f}ms", (10,60), self.font, 1, (0,0,255), 2, cv2.LINE_AA)
        self.guidisp = cv2.cvtColor(guidisp,cv2.COLOR_BGR2BGRA)
        return stop, steering_angle
    
    def detect(self):
        t1 = time.perf_counter()
        match self.linesrc:
            case 0: #Rectified R
                sourceL = self.remappedL
                sourceR = self.remappedR
            case 1: #Raw R
                sourceL = self.frameL
                sourceR = self.frameR #input img source
        
        #Create GUI overlay
        self.objdetectgui = np.zeros((self.gui_imgsz[1], self.gui_imgsz[0], 4), dtype=np.uint8) #BGRA
        
        if self.detect_imgsz != self.gui_imgsz: #If input imgsz != GUI imgsz
            #Calculate GUI scaling
            i_w, i_h = self.detect_imgsz
            o_w, o_h = self.gui_imgsz
            scale_w = o_w / i_w
            scale_h = o_h / i_h
            
            #Input image rescale
            h,w = sourceR.shape[:2] #Extract size
            if w>self.detect_imgsz[0] and h>self.detect_imgsz[1]: #Downscaling for faster speed
                self.detectresmode = 'AREA'
                #sourceL = cv2.resize(sourceL,self.detect_imgsz,interpolation=cv2.INTER_AREA)
                sourceR = cv2.resize(sourceR,self.detect_imgsz,interpolation=cv2.INTER_AREA)
            elif w<self.detect_imgsz[0] and h<self.detect_imgsz[1]: #Shitty pics need upscaling
                self.detectresmode = 'LANCZOS'
                #sourceL = cv2.resize(sourceL,self.detect_imgsz,interpolation=cv2.INTER_LANCZOS4)
                sourceR = cv2.resize(sourceR,self.detect_imgsz,interpolation=cv2.INTER_LANCZOS4)
            else:
                self.detectresmode = 'CUBIC'
                #sourceL = cv2.resize(sourceL,self.detect_imgsz,interpolation=cv2.INTER_CUBIC)
                sourceR = cv2.resize(sourceR,self.detect_imgsz,interpolation=cv2.INTER_CUBIC)
        
        else: #if input imgsz = GUI imgsz
            scale_w = 1
            scale_h = 1
            self.detectresmode = 'ORG'
        
        #resultL = self.net.predict(source=sourceL,verbose=False,conf=self.minconf)[0]
        results = self.net.predict(source=sourceR,verbose=False,conf=self.minconf)

        #results = [resultL,resultR]
        for result in results:
            boxes = result.boxes
            names = result.names
        
        detected_sign = []
        for box in boxes:
            objconf = float(box.conf.cpu())
            if objconf < self.minconf: continue
            objid = int(box.cls.cpu())
            x1,y1,x2,y2 = box.xyxy[0].cpu().numpy().astype(int)
            x1 = round(x1*scale_w)
            y1 = round(y1*scale_h)
            x2 = round(x2*scale_w)
            y2 = round(y2*scale_h)
            if x1 < 0: x1 = 0
            elif x1 > w : x1 = w
            if x2 < 0: x2 = 0
            elif x2 > w: x2 = w
            if y1 < 0: y1 = 0
            elif y1 > h: y1 = h
            if y2 < 0: y2 = 0
            elif y2 > h: y2 = h
            cv2.rectangle(self.objdetectgui, (x1,y1), (x2,y2), color=(0,255,0,255), thickness=2)
            label = names.get(objid)
            cv2.putText(self.objdetectgui, label,(x1,y1+20),self.font,1,(0,255,0,255),1,cv2.LINE_AA)
            detected_sign.append(label)
        
        t2 = time.perf_counter()
        self.detreactrate = (t2 - t1)*1000 #Reaction time in millisec
        
        speed_set = []
        for f in detected_sign:
            match f:
                case f if f == 'Stop' or f == 'Red Light': speed_set.append(0)
                case f if f == 'Speed limit (60km/h)' or f == 'Yellow Light': speed_set.append(1)
                case f if f == 'Speed limit (80km/h)': speed_set.append(2)
                case f if f == 'Speed Limit 100' or f == 'Green Light': speed_set.append(3)
        
        if not speed_set: return None
        else:
            return min(speed_set)
        
    def lanectrl(self,p):
        match self.linesrc:
            case 0: #Rectified R
                sourceL = self.remappedL
                sourceR = self.remappedR
            case 1: #Raw R
                source = self.frameL
                source = self.frameR #input img source

        #Create GUI overlay
        self.lanegui = np.zeros((self.gui_imgsz[1], self.gui_imgsz[0], 4), dtype=np.uint8) #BGRA
        t1 = time.perf_counter()
        # Convert the frame to HSV color space
        if self.lane_imgsz != self.gui_imgsz:
            i_w, i_h = self.lane_imgsz
            o_w, o_h = self.gui_imgsz
            scale_w = o_w / i_w
            scale_h = o_h / i_h
            #print(f"{scale_w},{scale_h}")
            lineimgL = cv2.resize(sourceL, (i_w,i_h), cv2.INTER_AREA)
            lineimgR = cv2.resize(sourceR, (i_w,i_h), cv2.INTER_AREA)
            
        else:
            scale_w = 1
            scale_h = 1
            lineimgL = sourceL
            lineimgR = sourceR
        
        hsvL = cv2.cvtColor(lineimgL, cv2.COLOR_BGR2HSV) #Convert HSV
        hsvR = cv2.cvtColor(lineimgR, cv2.COLOR_BGR2HSV) #Convert HSV

        # Define color ranges for yellow (lane lines) and black (middle of the road)
        yellow_lower = np.array([10, 40, 40], dtype=np.uint8)
        yellow_upper = np.array([50,255,255], dtype=np.uint8)
        black_lower = np.array([0, 0, 0], dtype=np.uint8)
        black_upper = np.array([180, 255, 30], dtype=np.uint8)

        # Create masks for yellow (lane) and black (middle) (Color filter)
        yellow_maskL = cv2.inRange(hsvL, yellow_lower, yellow_upper)
        black_maskL = cv2.inRange(hsvL, black_lower, black_upper)
        yellow_maskR = cv2.inRange(hsvR, yellow_lower, yellow_upper)
        black_maskR = cv2.inRange(hsvR, black_lower, black_upper)
        
        # Combine masks: Keep yellow lanes and remove the black middle
        combined_maskL = cv2.bitwise_and(yellow_maskL, cv2.bitwise_not(black_maskL))
        combined_maskR = cv2.bitwise_and(yellow_maskR, cv2.bitwise_not(black_maskR))
        #cv2.imshow('',combined_mask)
        
        #Equalizing histogram & Gaussian blur for smoothline
        combined_maskL = cv2.equalizeHist(combined_maskL)
        combined_maskR = cv2.equalizeHist(combined_maskR)
        combined_maskL = cv2.morphologyEx(combined_maskL, cv2.MORPH_CLOSE, np.ones((7,7), np.uint8)) #Apply morphologyEx to reconnect fragmented lines
        combined_maskR = cv2.morphologyEx(combined_maskR, cv2.MORPH_CLOSE, np.ones((7,7), np.uint8)) #Apply morphologyEx to reconnect fragmented lines
        #combined_maska = cv2.GaussianBlur(combined_mask, (7, 7), 1.5)
        #Dilate for bigger line recognition
        #combined_mask = cv2.erode(combined_maska, np.ones((5,5),np.uint8),4)
        
        
        # More sensitive Canny
        v_meanL = np.mean(combined_maskL)
        low = int(max(0, 0.6 * v_meanL))
        high = int(min(255, 1.3 * v_meanL))
        edgeL = cv2.Canny(combined_maskL, low, high, apertureSize=7, L2gradient=True)
        edgeL = cv2.equalizeHist(edgeL)
        
        v_meanR = np.mean(combined_maskR)
        low = int(max(0, 0.6 * v_meanR))
        high = int(min(255, 1.3 * v_meanR))
        edgeR = cv2.Canny(combined_maskR, low, high, apertureSize=7, L2gradient=True)
        edgeR = cv2.equalizeHist(edgeR)
        '''edges = cv2.dilate(edges, np.ones((5,5),np.uint8),0.5)
        edges = cv2.GaussianBlur(edges, (7, 7), 1.5)'''
        
        # Apply narrow ROI
        height, width = edgeR.shape
        ul = 0.25, 0.6
        ur = 0.75, 0.6
        ll = 0,1
        lr = 1, 1
        roi = np.array([[
            (0*ll[0], int(height * ll[1])),               # bottom left (start from lower area)
            (int(width * ul[0]), int(height * ul[1])),    # upper-left corner of trapezoid
            (int(width * ur[0]), int(height * ur[1])),    # upper-right corner
            (width*lr[0], int(height * lr[1]))            # bottom right
        ]], dtype=np.int32)
        roi_gui = np.array([[
            (int(0*ll[0]*scale_w), int(height * ll[1]*scale_h)),               # bottom left (start from lower area)
            (int(width * ul[0]*scale_w), int(height * ul[1]*scale_h)),    # upper-left corner of trapezoid
            (int(width * ur[0]*scale_w), int(height * ur[1]*scale_h)),    # upper-right corner
            (int(width*lr[0]*scale_w), int(height * lr[1]*scale_h))            # bottom right
        ]], dtype=np.int32)
        
        maskL = np.zeros_like(edgeL) #Filter out non-ROI region
        cv2.fillPoly(maskL, roi, 255)
        cropped_edgeL = cv2.bitwise_and(edgeL, maskL)
        cropped_edgeL = cv2.morphologyEx(cropped_edgeL, cv2.MORPH_CLOSE, np.ones((7,7), np.uint8)) #Apply morphologyEx to reconnect fragmented lines
        
        maskR = np.zeros_like(edgeR) #Filter out non-ROI region
        cv2.fillPoly(maskR, roi, 255)
        cropped_edgeR = cv2.bitwise_and(edgeR, maskR)
        cropped_edgeR = cv2.morphologyEx(cropped_edgeR, cv2.MORPH_CLOSE, np.ones((7,7), np.uint8)) #Apply morphologyEx to reconnect fragmented lines
        
        test = np.hstack((cropped_edgeL,cropped_edgeR))
        cv2.imshow('',test)
       
        
        lineL = cv2.HoughLinesP(
            cropped_edgeL,
            rho=1,
            theta=np.pi / 180,
            threshold=40,
            minLineLength=10,
            maxLineGap=30
        )
        
        lineR = cv2.HoughLinesP(
            cropped_edgeR,
            rho=1,
            theta=np.pi / 180,
            threshold=40,
            minLineLength=10,
            maxLineGap=30
        )
        
        lane_lines = []
        slope_l = []
        slope_r = []
        
        if lineL is not None:
            for line in lineL:
                x1, y1, x2, y2 = line[0]
                slope = (y2 - y1) / (x2 - x1) if x2 != x1 else float('inf')
                '''if 0.5 < abs(slope) < 2.0:  # Filter out near-horizontal lines
                    lane_lines.append((x1, y1, x2, y2))'''
                lane_lines.append((x1,y1,x2,y2))
                if slope > 0: slope_r.append(5/slope)
                elif slope < 0: slope_l.append(5/slope)
                else:
                    slope_l.append(0)
                    slope_r.append(0)
                    
        if lineR is not None:
            for line in lineR:
                x1, y1, x2, y2 = line[0]
                slope = (y2 - y1) / (x2 - x1) if x2 != x1 else float('inf')
                '''if 0.5 < abs(slope) < 2.0:  # Filter out near-horizontal lines
                    lane_lines.append((x1, y1, x2, y2))'''
                lane_lines.append((x1,y1,x2,y2))
                if slope > 0: slope_r.append(5/slope)
                elif slope < 0: slope_l.append(5/slope)
                else:
                    slope_l.append(0)
                    slope_r.append(0)
        
        if not slope_l: slope_l.append(0)
        if not slope_r: slope_r.append(0)
        #print(slope_l)
        #print(slope_r)
        #Compare amount of lines
        if len(slope_l) > len(slope_r): final_angle = min(slope_l)
        else: final_angle = max(slope_r)
                
                
        
        # Separate left and right lanes
        left_x, right_x = [], []
        for x1, y1, x2, y2 in lane_lines:
            if (x1 + x2) / 2 < width // 2:  # Left lane
                left_x.append((x1 + x2) / 2)
            else:  # Right lane
                right_x.append((x1 + x2) / 2)

        if left_x and right_x:
            lane_center = (min(left_x) + max(right_x)) / 2
        elif left_x:
            lane_center = min(left_x) + 100  # Assume a standard lane width
        elif right_x:
            lane_center = max(right_x) - 100
        else:
            lane_center = width // 2  # Default to center

        # Calculate steering angle
        frame_center = width // 2
        #self.steering_angle = (lane_center - frame_center) * p  # Proportional control constant
        self.steering_angle = int(final_angle)*p
        #print(self.steering_angle)
        '''if not slope_list: self.steering_angle = 0
        else:
            slope_list = np.array(slope_list)
            slope_list = cv2.normalize(slope_list, None, -128, 127, cv2.NORM_MINMAX)
            slope_list = np.int8(slope_list) 
            average = np.mean(slope_list)
            self.steering_angle = average.astype(np.int8)'''
            
        # Draw lane lines and center point
        cv2.circle(self.lanegui, (int(lane_center*scale_w), int((height - 20)*scale_h)), 10, (0, 0, 255, 255), -1)
        for x1, y1, x2, y2 in lane_lines:
            cv2.line(self.lanegui, (int(x1*scale_w), int(y1*scale_h)), (int(x2*scale_w), int(y2*scale_h)), (0, 255, 0, 255), 5)

        t2 = time.perf_counter()
        self.lanereactrate = (t2 - t1)*1000 #Reaction time in millisec
        cv2.polylines(self.lanegui, roi_gui , isClosed=True, color=(0, 255, 0, 255), thickness=2)
        return self.steering_angle
        
    def close(self):
        self.camL.stop()
        self.camR.stop()
        
    def trafficlight(self):
        #Create overlays for GUI
        self.tlightgui = np.zeros((self.gui_imgsz[1], self.gui_imgsz[0], 4), dtype=np.uint8) #BGRA
        
        # Convert to HSV
        h, w, _ = self.frameR.shape
        hsv = cv2.cvtColor(self.frameR, cv2.COLOR_BGR2HSV)

        # Define color ranges in HSV
        # Red can wrap around HSV hue (0° and 180°), so we use two ranges
        red_lower1 = np.array([0, 100, 100])
        red_upper1 = np.array([10, 255, 255])
        red_lower2 = np.array([160, 100, 100])
        red_upper2 = np.array([180, 255, 255])

        yellow_lower = np.array([15, 120, 120])
        yellow_upper = np.array([35, 255, 255])

        green_lower = np.array([40, 70, 70])
        green_upper = np.array([90, 255, 255])

        # Create masks for each color
        mask_red1 = cv2.inRange(hsv, red_lower1, red_upper1)
        mask_red2 = cv2.inRange(hsv, red_lower2, red_upper2)
        mask_red = cv2.bitwise_or(mask_red1, mask_red2)

        mask_yellow = cv2.inRange(hsv, yellow_lower, yellow_upper)
        mask_green = cv2.inRange(hsv, green_lower, green_upper)

        # Use morphological operations to clean up noise
        kernel = np.ones((5, 5), np.uint8)
        mask_red = cv2.morphologyEx(mask_red, cv2.MORPH_OPEN, kernel)
        mask_yellow = cv2.morphologyEx(mask_yellow, cv2.MORPH_OPEN, kernel)
        mask_green = cv2.morphologyEx(mask_green, cv2.MORPH_OPEN, kernel)

        # Count non-zero pixels for each color
        red_area = cv2.countNonZero(mask_red)
        yellow_area = cv2.countNonZero(mask_yellow)
        green_area = cv2.countNonZero(mask_green)

        # Determine active color
        max_area = max(red_area, yellow_area, green_area)
        color = "UNKNOWN"
        speed = 2
        if max_area > 200:   # adjust threshold for your setup
            if max_area == red_area:
                color = "RED"
                speed = 0
            elif max_area == yellow_area:
                color = "YELLOW"
                speed = 1
            else:
                color = "GREEN"
                speed = 2
        
        # Show masks (for debugging)
        '''cv2.imshow("Red Mask", mask_red)
        cv2.imshow("Yellow Mask", mask_yellow)
        cv2.imshow("Green Mask", mask_green)'''
        # Display detected color on the frame
        x,y = 0,35
        '''(tw,th),bl = cv2.getTextSize(f"Detected: {color}", self.font, 0.5, 1)
        cv2.rectangle(self.tlightgui, (x, y - th - bl),
        (x + tw, y + bl), color=(0,0,0,255), thickness=-1)'''
        cv2.putText(self.tlightgui, f"Detected: {color}", (x,y),
                    self.font, 0.5, (0, 255, 0, 255), 1, cv2.LINE_AA)
        return speed
        

