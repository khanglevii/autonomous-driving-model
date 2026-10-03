import cv2
import numpy as np
from picamera2 import Picamera2

# Load saved maps
data = np.load("assets/calib/stereo_maps.npz")
mapLx, mapLy, mapRx, mapRy = [data[f] for f in ("mapLx","mapLy","mapRx","mapRy")]

# Init cameras
picamL = Picamera2(1)
picamR = Picamera2(0)

configL = picamL.create_preview_configuration(main={"size": (640,480)})
configR = picamR.create_preview_configuration(main={"size": (640,480)})
picamL.configure(configL)
picamR.configure(configR)
picamL.start()
picamR.start()

while True:
    frameL = picamL.capture_array()
    frameR = picamR.capture_array()
    
    frameL = cv2.cvtColor(frameL, cv2.COLOR_RGB2BGR)
    frameR = cv2.cvtColor(frameR, cv2.COLOR_RGB2BGR)

    rectL = cv2.remap(frameL, mapLx, mapLy, cv2.INTER_LINEAR)
    rectR = cv2.remap(frameR, mapRx, mapRy, cv2.INTER_LINEAR)

    vis = np.hstack((rectL, rectR))
    cv2.imshow("Rectified Stereo", vis)

    if cv2.waitKey(1) == 27:  # ESC
        break

cv2.destroyAllWindows()
picamL.stop()
picamR.stop()
