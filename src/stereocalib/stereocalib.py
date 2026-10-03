import cv2
import numpy as np
from picamera2 import Picamera2
import time
import glob
import os

def calib(inputfolder='',map_file='',calres=(640,480)):
    # Step 2: Detect chessboard corners
    print(f"Starting calibration at {inputfolder}")
    
    objp = np.zeros((np.prod(chessboard_size), 3), np.float32)
    objp[:, :2] = np.indices(chessboard_size).T.reshape(-1, 2)
    objp *= square_size
    
    objpoints = []
    imgpointsL = []
    imgpointsR = []
    
    images_left = sorted(glob.glob(f"{inputfolder}/left_*.png"))
    images_right = sorted(glob.glob(f"{inputfolder}/right_*.png"))
    
    for imgL, imgR in zip(images_left, images_right):
        grayL = cv2.imread(imgL, cv2.IMREAD_GRAYSCALE)
        grayR = cv2.imread(imgR, cv2.IMREAD_GRAYSCALE)
    
        retL, cornersL = cv2.findChessboardCorners(grayL, chessboard_size, None)
        retR, cornersR = cv2.findChessboardCorners(grayR, chessboard_size, None)
    
        if retL and retR:
            objpoints.append(objp)
            imgpointsL.append(cornersL)
            imgpointsR.append(cornersR)
    
    print(f"Found corners in {len(objpoints)} pairs of images.")
    
    # Step 3: Calibrate each camera
    retL, mtxL, distL, _, _ = cv2.calibrateCamera(objpoints, imgpointsL, grayL.shape[::-1], None, None)
    retR, mtxR, distR, _, _ = cv2.calibrateCamera(objpoints, imgpointsR, grayR.shape[::-1], None, None)
    
    # Step 4: Stereo calibration
    flags = cv2.CALIB_FIX_INTRINSIC
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 1e-6)
    
    ret, _, _, _, _, R, T, E, F = cv2.stereoCalibrate(
        objpoints, imgpointsL, imgpointsR,
        mtxL, distL, mtxR, distR, grayL.shape[::-1],
        criteria=criteria, flags=flags
    )
    
    # Step 5: Stereo rectify
    R1, R2, P1, P2, Q, _, _ = cv2.stereoRectify(
        mtxL, distL, mtxR, distR, grayL.shape[::-1], R, T
    )
    
    mapLx, mapLy = cv2.initUndistortRectifyMap(mtxL, distL, R1, P1, grayL.shape[::-1], cv2.CV_32FC1)
    mapRx, mapRy = cv2.initUndistortRectifyMap(mtxR, distR, R2, P2, grayR.shape[::-1], cv2.CV_32FC1)
    
    # Step 7: Save everything
    np.savez(map_file,
             mapLx=mapLx, mapLy=mapLy,
             mapRx=mapRx, mapRy=mapRy,
             Q=Q,calres=calres)
    
    print(f"Calibration maps, Q matrix, and LUT saved to {map_file}")
    
# ---------- SETTINGS ----------
linecalres = 360, 270
stereocalres = 160, 120 #Calibration resolution (will be used for inference, WxH)
chessboard_size = (8, 6)   # inner corners of your chessboard
square_size = 0.03         # size of a square in meters (adjust to your chessboard)
num_pairs = 50             # number of image pairs to capture
save_folder = "img"
stereomapfile = "assets/calib/stereo_maps.npz"
linemapfile = "assets/calib/line_maps.npz"
# ------------------------------
cv2.setNumThreads(1)
# Create save folder if missing
os.makedirs(save_folder, exist_ok=True)
os.makedirs(f"{save_folder}/resized", exist_ok=True)
os.makedirs(f"{save_folder}/org", exist_ok=True)

# Step 1: Capture images from both cameras
print("Initializing cameras...")
picamL = Picamera2(1)
picamR = Picamera2(0)

configL = picamL.create_preview_configuration(main={"size": (640, 480)})
configR = picamR.create_preview_configuration(main={"size": (640, 480)})
picamL.configure(configL)
picamR.configure(configR)
picamL.start()
picamR.start()

time.sleep(2)  # warm-up time

print(f"Press SPACE to capture {num_pairs} pairs of images, ESC to quit.")


i = 0
while i < num_pairs:
    frameL = picamL.capture_array()
    frameR = picamR.capture_array()
    frameL = cv2.cvtColor(frameL, cv2.COLOR_RGBA2BGR)
    frameR = cv2.cvtColor(frameR, cv2.COLOR_RGBA2BGR)
    green_scale = 0.91
    frameL[:, :, 1] = np.clip(frameL[:, :, 1] * green_scale, 0, 255).astype(np.uint8)
    green_scale = 0.82
    frameR[:, :, 1] = np.clip(frameR[:, :, 1] * green_scale, 0, 255).astype(np.uint8)
    vis = np.hstack((frameL, frameR))
    cv2.imshow("Stereo Capture (Left | Right)", vis)

    key = cv2.waitKey(1)
    if key == 32:  # SPACE
        frameL = cv2.resize(frameL,linecalres,cv2.INTER_AREA)
        frameR = cv2.resize(frameR,linecalres,cv2.INTER_AREA)
        cv2.imwrite(f"{save_folder}/org/left_{i}.png", frameL)
        cv2.imwrite(f"{save_folder}/org/right_{i}.png", frameR)
        frameL = cv2.resize(frameL,stereocalres,cv2.INTER_AREA)
        frameR = cv2.resize(frameR,stereocalres,cv2.INTER_AREA)
        cv2.imwrite(f"{save_folder}/resized/left_{i}.png", frameL)
        cv2.imwrite(f"{save_folder}/resized/right_{i}.png", frameR)
        print(f"Saved pair {i}")
        i += 1
    elif key == 27:  # ESC
        break

cv2.destroyAllWindows()
picamL.stop()
picamR.stop()

calib("img/resized",stereomapfile,stereocalres)
calib("img/org",linemapfile,linecalres)
    
