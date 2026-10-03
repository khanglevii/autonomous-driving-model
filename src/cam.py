from picamera2 import Picamera2

# List all detected cameras
cameras = Picamera2.global_camera_info()
for idx, cam in enumerate(cameras):
    print(f"Index {idx}: {cam}")
