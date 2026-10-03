        import cv2
import numpy as np

# --- Stereo setup ---
stereo = cv2.StereoBM_create(numDisparities=96, blockSize=15)  # tune these values

# --- Camera capture ---
camL = cv2.VideoCapture(0)
camR = cv2.VideoCapture(1)

while True:
    retL, frameL = camL.read()
    retR, frameR = camR.read()
    if not (retL and retR):
        break

    # Convert to grayscale
    grayL = cv2.cvtColor(frameL, cv2.COLOR_BGR2GRAY)
    grayR = cv2.cvtColor(frameR, cv2.COLOR_BGR2GRAY)

    # Compute disparity map
    disparity = stereo.compute(grayL, grayR)
    disp = cv2.normalize(disparity, None, 0, 255, cv2.NORM_MINMAX)
    disp = np.uint8(disp)

    # --- Threshold to find close obstacles ---
    _, mask = cv2.threshold(disp, 180, 255, cv2.THRESH_BINARY)

    # Divide into 3 vertical zones
    h, w = mask.shape
    left_zone = mask[:, :w//3]
    mid_zone = mask[:, w//3:2*w//3]
    right_zone = mask[:, 2*w//3:]

    # Calculate density of white pixels (closer = more disparity)
    left_density = cv2.countNonZero(left_zone)
    mid_density = cv2.countNonZero(mid_zone)
    right_density = cv2.countNonZero(right_zone)

    # --- Decision logic ---
    turn = "forward"
    if left_density > mid_density and left_density > right_density and left_density > 500:
        turn = "turn right"
    elif mid_density > left_density and mid_density > right_density and mid_density > 500:
        turn = "turn left"
    elif right_density > left_density and right_density > mid_density and right_density > 500:
        turn = "turn left"  # or stop, up to your choice

    # Draw regions
    cv2.line(frameL, (w//3, 0), (w//3, h), (0, 255, 255), 2)
    cv2.line(frameL, (2*w//3, 0), (2*w//3, h), (0, 255, 255), 2)
    cv2.putText(frameL, f"Decision: {turn}", (20, 40),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

    cv2.imshow("Left Camera", frameL)
    cv2.imshow("Depth Map", disp)
    cv2.imshow("Obstacle Mask", mask)

    key = cv2.waitKey(1)
    if key == 27:
        break

camL.release()
camR.release()
cv2.destroyAllWindows()







