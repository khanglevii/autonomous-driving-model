def lanectrl(self,p):
        match self.linesrc:
            case 0: #Rectified R
                source = self.remappedR
            case 1: #Raw R
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
            lineimg = cv2.resize(source, (i_w,i_h), cv2.INTER_AREA)
            
        else:
            scale_w = 1
            scale_h = 1
            lineimg = source
        
        hsv = cv2.cvtColor(lineimg, cv2.COLOR_BGR2HSV) #Convert HSV
        

        # Define color ranges for yellow (lane lines) and black (middle of the road)
        yellow_lower = np.array([10, 40, 40], dtype=np.uint8)
        yellow_upper = np.array([50,255,255], dtype=np.uint8)
        black_lower = np.array([0, 0, 0], dtype=np.uint8)
        black_upper = np.array([180, 255, 30], dtype=np.uint8)

        # Create masks for yellow (lane) and black (middle) (Color filter)
        yellow_mask = cv2.inRange(hsv, yellow_lower, yellow_upper)
        black_mask = cv2.inRange(hsv, black_lower, black_upper)
        #cv2.imshow('',hsv)
        cv2.imshow('yellow',yellow_mask)
        #cv2.imshow('black',black_mask)
        # Combine masks: Keep yellow lanes and remove the black middle
        combined_mask = cv2.bitwise_and(yellow_mask, cv2.bitwise_not(black_mask))
        #cv2.imshow('',combined_mask)
        
        #Gaussian filter for denoise
        combined_mask = cv2.equalizeHist(combined_mask)
        # Slight blur for continuity
        combined_mask = cv2.GaussianBlur(combined_mask, (7, 7), 1.5)
    
        # More sensitive Canny
        v_mean = np.mean(combined_mask)
        low = int(max(0, 0.6 * v_mean))
        high = int(min(255, 1.3 * v_mean))
        edges = cv2.Canny(combined_mask, low, high, apertureSize=3, L2gradient=True)
        
        # Apply narrow ROI
        height, width = edges.shape
        ul = 0.35, 0.3
        ur = 0.65, 0.3
        ll = 0, 0.7
        lr = 1, 0.7
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
        mask = np.zeros_like(edges)
        cv2.fillPoly(mask, roi, 255)
        cropped_edges = cv2.bitwise_and(edges, mask)
        kernel = np.ones((5,5), np.uint8)
        cropped_edges = cv2.morphologyEx(cropped_edges, cv2.MORPH_CLOSE, kernel)
        #cv2.imshow('canny',cropped_edges)

        # Sensitive Hough for short, faint lines
        lines = cv2.HoughLinesP(
            cropped_edges,
            rho=1,
            theta=np.pi / 180,
            threshold=20,
            minLineLength=5,
            maxLineGap=30
        )

        lane_lines = []
        if lines is not None:
            for line in lines:
                x1, y1, x2, y2 = line[0]
                slope = (y2 - y1) / (x2 - x1) if x2 != x1 else float('inf')
                '''if 0.5 < abs(slope) < 2.0:  # Filter out near-horizontal lines
                    lane_lines.append((x1, y1, x2, y2))'''
                if slope != float('inf'): 
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
        self.steering_angle = (lane_center - frame_center) * p  # Proportional control constant

        # Draw lane lines and center point
        cv2.circle(self.lanegui, (int(lane_center*scale_w), int((height - 20)*scale_h)), 10, (0, 0, 255, 255), -1)
        for x1, y1, x2, y2 in lane_lines:
            cv2.line(self.lanegui, (int(x1*scale_w), int(y1*scale_h)), (int(x2*scale_w), int(y2*scale_h)), (0, 255, 0, 255), 5)

        # Display masks for debugging
        #cv2.imshow('Yellow Mask', yellow_mask)
        #cv2.imshow('Black Mask', black_mask)
        #cv2.imshow('Combined Mask', combined_mask)
        
        x,y = 0,13
        '''(tw,th),bl = cv2.getTextSize(f"Angle: {self.steering_angle:.2f}", self.font, 0.5, 1)
        cv2.rectangle(self.lanegui, (x, y - th - bl),
        (x + tw, y + bl), color=(0,0,0,255), thickness=-1)'''
        '''cv2.putText(self.lanegui, f"Angle: {self.steering_angle:.2f}", (x,y),
                    self.font, 0.5, (127, 255, 163, 255), 1, cv2.LINE_AA)'''
        '''cv2.polylines(hsv, roi , isClosed=True, color=(0, 255, 0, 255), thickness=2)
        cv2.imshow('',hsv)'''
        cv2.polylines(self.lanegui, roi_gui , isClosed=True, color=(0, 255, 0, 255), thickness=2)
        t2 = time.perf_counter()
        self.lanereactrate = (t2 - t1)*1000 #Reaction time in millisec
        '''h,w,_ = self.lanegui.shape
        cv2.putText(self.lanegui,f"Lanerate: {rate:.2f}ms",(w-170,26),self.font,0.5,(0,0,255,255),1,cv2.LINE_AA)'''
        return self.steering_angle
