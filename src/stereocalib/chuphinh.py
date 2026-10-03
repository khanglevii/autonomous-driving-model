import cv2
import numpy as np
from picamera2 import Picamera2
import time
import glob
import os

save_folder = "img"
start = 1
map_file = "assets/calib/stereo_maps.npz"
# ------------------------------

# Create save folder if missing
os.makedirs(save_folder, exist_ok=True)

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

print(f"Press SPACE to images, ESC to quit.")

i = start
while True:
    #frameL = picamL.capture_array()
    frameR = picamR.capture_array()
    #frameL = cv2.cvtColor(frameL, cv2.COLOR_RGBA2BGR)
    frameR = cv2.cvtColor(frameR, cv2.COLOR_RGBA2BGR)
    #vis = np.hstack((frameL, frameR))
    cv2.imshow("Stereo Capture (Left | Right)", frameR)

    key = cv2.waitKey(1)
    if key == 32:  # SPACE
        cv2.imwrite(f"{save_folder}/right_{i}.png", frameR)
        i += 1
    elif key == 27:  # ESC
        break

cv2.destroyAllWindows()
picamL.stop()
picamR.stop()
