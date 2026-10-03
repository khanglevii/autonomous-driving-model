import cv2
import numpy as np

def detect_lane_lines(frame):
    """
    Detects two yellow lane markings (left and right) with a black middle
    and calculates the steering direction.

    Args:
        frame (ndarray): The input video frame.

    Returns:
        steering_angle (float): The angle to steer the car toward the lane center.
        lanes_overlay (ndarray): The frame with lane lines overlaid.
    """
    # Convert the frame to HSV color space
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

    # Define color ranges for yellow (lane lines) and black (middle of the road)
    yellow_lower = np.array([18, 100, 100], dtype=np.uint8)
    yellow_upper = np.array([35, 255, 255], dtype=np.uint8)
    black_lower = np.array([0, 0, 0], dtype=np.uint8)
    black_upper = np.array([180, 255, 70], dtype=np.uint8)

    # Create masks for yellow (lane) and black (middle)
    yellow_mask = cv2.inRange(hsv, yellow_lower, yellow_upper)
    black_mask = cv2.inRange(hsv, black_lower, black_upper)

    # Combine masks: Keep yellow lanes and remove the black middle
    combined_mask = cv2.bitwise_and(yellow_mask, cv2.bitwise_not(black_mask))

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
    lanes_overlay = frame.copy()
    for x1, y1, x2, y2 in lane_lines:
        cv2.line(lanes_overlay, (x1, y1), (x2, y2), (0, 255, 0), 5)
    cv2.circle(lanes_overlay, (int(lane_center), height - 20), 10, (0, 0, 255), -1)

    # Display masks for debugging
    #cv2.imshow('Yellow Mask', yellow_mask)
    #cv2.imshow('Black Mask', black_mask)
    #cv2.imshow('Combined Mask', combined_mask)

    return steering_angle, lanes_overlay
