import cv2
import numpy as np

def detect_traffic_light_color(frame):
    # Resize for faster processing
    frame_resized = cv2.resize(frame, (640, 480))

    # Convert to HSV
    hsv = cv2.cvtColor(frame_resized, cv2.COLOR_BGR2HSV)

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
    if max_area > 500:   # adjust threshold for your setup
        if max_area == red_area:
            color = "RED"
        elif max_area == yellow_area:
            color = "YELLOW"
        else:
            color = "GREEN"

    # Display detected color on the frame
    cv2.putText(frame_resized, f"Detected: {color}", (20, 40),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)

    # Show masks (for debugging)
    cv2.imshow("Red Mask", mask_red)
    cv2.imshow("Yellow Mask", mask_yellow)
    cv2.imshow("Green Mask", mask_green)

    return frame_resized


def main():
    cap = cv2.VideoCapture(0)  # Change 0 to video file path if needed

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        output_frame = detect_traffic_light_color(frame)
        cv2.imshow("Traffic Light Detection", output_frame)

        if cv2.waitKey(1) & 0xFF == 27:  # Press ESC to exit
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()

