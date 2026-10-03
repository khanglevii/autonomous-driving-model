import cv2
import time
import numpy as np
from picamera2 import Picamera2
from ultralytics import YOLO

class vision:
    def __init__(self,camL,camR,calib_file,model_path,minconf,lane_imgsz,i_imgsz,o_imgsz=None):
         #Imgsz standard: H,W
        self.lane_imgsz = lane_imgsz[1], lane_imgsz[0]
        self.i_imgsz = i_imgsz[1], i_imgsz[0]
        if o_imgsz != None: #Set default output size
            self.o_imgsz = o_imgsz[1], o_imgsz[0]
        else: self.o_imgsz = 480,640
        
        self.minconf = minconf
        self.font = cv2.FONT_HERSHEY_DUPLEX
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
        stereo_cal = np.load(calib_file)
        self.mapLx, self.mapLy, self.mapRx, self.mapRy, self.Q, self.calres = [stereo_cal[f] for f in ("mapLx","mapLy","mapRx","mapRy","Q","calres")]
        
        # StereoBM parameters
        num_disp = 96   # must be multiple of 16
        block_size = 5 # must be odd, usually 5-21 works best

        '''self.stereo = cv2.StereoBM_create(
            numDisparities=num_disp,
            blockSize=block_size
        )
        self.stereo.setPreFilterCap(31)
        self.stereo.setTextureThreshold(5)     # lower, to keep weakly-textured areas
        self.stereo.setUniquenessRatio(10)
        self.stereo.setSpeckleWindowSize(50)
        self.stereo.setSpeckleRange(32)'''
        
        # StereoSGBM parameters
        window_size = 5
        #num_disp = 96  # must be divisible by 16
        self.stereo = cv2.StereoSGBM_create(
            minDisparity=0,
            numDisparities=num_disp,
            blockSize=block_size,
            P1=8 * 3 * window_size**2,
            P2=32 * 3 * window_size**2,
            disp12MaxDiff=1,
            uniquenessRatio=10,
            speckleWindowSize=50,
            speckleRange=32
        )
        
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
        
        self.gui = cv2.cvtColor(self.frameR, cv2.COLOR_BGR2BGRA)
        h,w,_ = self.frameR.shape
        if  (h,w) != self.o_imgsz: self.gui = cv2.resize(self.frameR, (self.o_imgsz[1],self.o_imgsz[0]), cv2.INTER_NEAREST)
        
    def maingui(self):
        result = self.gui.astype(np.float32)
        overlays = [self.tlightgui, self.objdetectgui, self.lanegui]

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
        cv2.imshow("Main GUI", result)
        
    def depth_est(self): #Depth estimation
        t1 = time.perf_counter()
        #Downscale for faster speed
        rectL = cv2.resize(self.frameL,self.calres,interpolation=cv2.INTER_NEAREST)
        rectR = cv2.resize(self.frameR,self.calres,interpolation=cv2.INTER_NEAREST)
        
        #Stereo cameras calibration
        rectL = cv2.remap(rectL, self.mapLx, self.mapLy, cv2.INTER_NEAREST)
        rectR = cv2.remap(rectR, self.mapRx, self.mapRy, cv2.INTER_NEAREST)
        
        # Grayscale for disparity
        grayL = cv2.cvtColor(rectL, cv2.COLOR_BGR2GRAY)
        grayR = cv2.cvtColor(rectR, cv2.COLOR_BGR2GRAY)
        
        #Preprocessing steps
        grayL = cv2.equalizeHist(grayL)
        grayR = cv2.equalizeHist(grayR)
        
        blur=5,5
        grayL = cv2.GaussianBlur(grayL, blur, 1)
        grayR = cv2.GaussianBlur(grayR, blur, 1)
        
        # Compute disparity
        disparity = self.stereo.compute(grayL, grayR).astype(np.float32) / 16.0 #Divide 16 since OPENCV store disparity as int16, by calculate in x16.
        
        #Postprocessing steps
        disparity = cv2.medianBlur(disparity, 5) #Apply median blur
        #disparity = cv2.bilateralFilter(disparity, 9, 75, 75)
        kernel = np.ones((5,5),np.uint8)
        disparity = cv2.morphologyEx(disparity, cv2.MORPH_CLOSE, kernel)
        
        # Reproject to 3D
        points_3D = cv2.reprojectImageTo3D(disparity, self.Q)
        
        # Get center point
        h, w = disparity.shape
        cx, cy = w // 2, h // 2
        
        # ROI instead of single pixel (for stability)
        roi_size = 5
        roi = points_3D[cy - roi_size:cy + roi_size, cx - roi_size:cx + roi_size, :]

        # Extract Z (depth)
        Z = roi[:, :, 2]
        Z = Z[np.isfinite(Z)]  # remove invalid values (inf/nan)
        
        distance = float(np.median(Z))
        if distance < 0: distance = np.nan
        # Draw crosshair
        guidisp = rectL
        '''h,w,_ = guidisp.shape
        cx = w//2
        cy = h//2'''
        cv2.drawMarker(guidisp, (cx, cy), (0,0,255), cv2.MARKER_CROSS, 20, 2)
        # Normalize disparity for display
        disp_vis = cv2.normalize(disparity, None, 0, 255, cv2.NORM_MINMAX)
        disp_vis = np.uint8(disp_vis)
        disp_vis = cv2.resize(disp_vis,(self.frameL.shape[1],self.frameL.shape[0]),interpolation=cv2.INTER_NEAREST)
        guidisp = cv2.resize(guidisp,(self.frameL.shape[1],self.frameL.shape[0]),interpolation=cv2.INTER_NEAREST)
        guidisp = np.hstack((guidisp, cv2.cvtColor(disp_vis, cv2.COLOR_GRAY2BGR)))
        t2 = time.perf_counter()
        rate = (t2 - t1)*1000 #Reaction time in millisec
        #Txtbox
        cv2.rectangle(guidisp, (0,0), (cx,70), (0,0,0), -1)
        # Overlay distance text
        cv2.putText(guidisp, f"Distance: {distance:.2f} m", (10,30),
                self.font, 1, (0,255,0), 2, cv2.LINE_AA)
        cv2.putText(guidisp, f"Rate: {rate:.2f}ms", (10,60), self.font, 1, (0,0,255), 2, cv2.LINE_AA)
        cv2.putText(guidisp, f"CalRes: {self.calres}", (10,80), self.font, 0.5, (0,255,127), 2, cv2.LINE_AA)
        cv2.imshow("Distance", guidisp)
        
    def trafficlight(self):
        #Create overlays for GUI
        self.tlightgui = np.zeros((self.o_imgsz[0], self.o_imgsz[1], 4), dtype=np.uint8) #BGRA
        
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
        x,y = 0,40
        (tw,th),bl = cv2.getTextSize(f"Detected: {color}", self.font, 0.5, 1)
        cv2.rectangle(self.tlightgui, (x, y - th - bl),
        (x + tw, y + bl), color=(0,0,0,255), thickness=-1)
        cv2.putText(self.tlightgui, f"Detected: {color}", (x,y),
                    self.font, 0.5, (0, 255, 0, 255), 1, cv2.LINE_AA)
        return speed
    
    def detect(self):
        #Create GUI overlay
        self.objdetectgui = np.zeros((self.o_imgsz[0], self.o_imgsz[1], 4), dtype=np.uint8) #BGRA
        
        t1 = time.perf_counter()
        if self.i_imgsz != self.o_imgsz:
            i_h, i_w = self.i_imgsz
            o_h, o_w = self.o_imgsz
            scale_w = o_w / i_w
            scale_h = o_h / i_h
            objimg = cv2.resize(self.frameR, (self.i_imgsz[1], self.i_imgsz[0]), cv2.INTER_NEAREST)
        else:
            scale_w = 1
            scale_h = 1
            objimg = self.frameR
        
        results = self.net.predict(source=objimg,verbose=False,conf=self.minconf)

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
            cv2.rectangle(self.objdetectgui, (x1,y1), (x2,y2), color=(32,128,143,255), thickness=1)
            label = names.get(objid)
            cv2.putText(self.objdetectgui, label,(x1,y1-10),self.font,1,(0,255,0,255),1,cv2.LINE_AA)
            detected_sign.append(label)
            
        t2 = time.perf_counter()
        '''rate = (t2 - t1)*1000 #Reaction time in millisec
        #Txtbox
        cv2.rectangle(objimg, (0,0), (256,40), (0,0,0), -1)
        cv2.putText(objimg, f"Rate: {rate:.2f}ms", (10,30), self.font, 1.8, (0,255,0), 2, cv2.LINE_AA)'''
        
    def lanectrl(self):
        #Create GUI overlay
        self.lanegui = np.zeros((self.o_imgsz[0], self.o_imgsz[1], 4), dtype=np.uint8) #BGRA
        
        t1 = time.perf_counter()
        # Convert the frame to HSV color space
        if self.lane_imgsz != self.o_imgsz:
            i_h, i_w = self.i_imgsz
            o_h, o_w = self.o_imgsz
            scale_w = o_w / i_w
            scale_h = o_h / i_h
            lineimg = cv2.resize(self.frameR, (self.lane_imgsz[1],self.lane_imgsz[0]), cv2.INTER_NEAREST)
        else:
            scale_w = 1
            scale_h = 1
            lineimg = self.frameR
            
        hsv = cv2.cvtColor(lineimg, cv2.COLOR_BGR2HSV)

        # Define color ranges for yellow (lane lines) and black (middle of the road)
        yellow_lower = np.array([18, 100, 100], dtype=np.uint8)
        yellow_upper = np.array([35, 255, 255], dtype=np.uint8)
        black_lower = np.array([0, 0, 0], dtype=np.uint8)
        black_upper = np.array([180, 255, 70], dtype=np.uint8)

        # Create masks for yellow (lane) and black (middle) (Color filter)
        yellow_mask = cv2.inRange(hsv, yellow_lower, yellow_upper)
        black_mask = cv2.inRange(hsv, black_lower, black_upper)

        # Combine masks: Keep yellow lanes and remove the black middle
        combined_mask = cv2.bitwise_and(yellow_mask, cv2.bitwise_not(black_mask))
            
        #Gaussian filter for denoise
        combined_mask = cv2.equalizeHist(combined_mask)
        blur=9,9
        combined_mask = cv2.GaussianBlur(combined_mask, blur, 3)
        # Edge detection
        edges = cv2.Canny(combined_mask, 50, 150)

        # Region of interest (ROI)
        height, width = edges.shape
        roi = np.array([[
            (0, height),
            (width // 2 - 100, height // 2),
            (width // 2 + 100, height // 2),
            (width, height)
        ]], dtype=np.int32)
        mask = np.zeros_like(edges)
        cv2.fillPoly(mask, roi, 255)
        cropped_edges = cv2.bitwise_and(edges, mask)

        # Detect lines using Hough Transform
        lines = cv2.HoughLinesP(cropped_edges, 1, np.pi / 180, 50, minLineLength=50, maxLineGap=150)
        lane_lines = []
        if lines is not None:
            for line in lines:
                x1, y1, x2, y2 = line[0]
                slope = (y2 - y1) / (x2 - x1) if x2 != x1 else float('inf')
                if 0.5 < abs(slope) < 2.0:  # Filter out near-horizontal lines
                    lane_lines.append((x1, y1, x2, y2))

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
        steering_angle = (lane_center - frame_center) * 0.1  # Proportional control constant

        # Draw lane lines and center point
        cv2.circle(self.lanegui, (int(lane_center), height - 20), 10, (0, 0, 255, 255), -1)
        for x1, y1, x2, y2 in lane_lines:
            cv2.line(self.lanegui, (x1, y1), (x2, y2), (0, 255, 0, 255), 5)

        # Display masks for debugging
        #cv2.imshow('Yellow Mask', yellow_mask)
        #cv2.imshow('Black Mask', black_mask)
        #cv2.imshow('Combined Mask', combined_mask)
        
        x,y = 0,10
        (tw,th),bl = cv2.getTextSize(f"Angle: {steering_angle:.2f}", self.font, 0.5, 1)
        cv2.rectangle(self.lanegui, (x, y - th - bl),
        (x + tw, y + bl), color=(0,0,0,255), thickness=-1)
        cv2.putText(self.lanegui, f"Angle: {steering_angle:.2f}", (x,y),
                    self.font, 0.5, (127, 255, 163, 255), 1, cv2.LINE_AA)
                    
        t2 = time.perf_counter()
        '''rate = (t2 - t1)*1000 #Reaction time in millisec
        
        #Txtbox
        cv2.rectangle(lanes_overlay, (0,0), (256,40), (0,0,0), -1)
        cv2.putText(lanes_overlay, f"Rate: {rate:.2f}ms", (10,30), self.font, 1.8, (0,255,0), 2, cv2.LINE_AA)'''
        return steering_angle
        
    def close(self):
        self.camL.stop()
        self.camR.stop()
        

