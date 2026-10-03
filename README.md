# Autonomous Driving Model

A small self-driving car project for a Raspberry Pi. It combines **lane detection**
(Hough transform), **traffic/speed-sign detection** (YOLO), and **stereo cameras**
for distance estimation, and drives the motors through GPIO.

## Features
- Lane following using a Hough-transform-based detector (`houghalgor.py`)
- Traffic and speed-sign detection with YOLO (v3 / v5 / v8 / v11) via Ultralytics or OpenCV DNN
- Multiple inference formats: ONNX, NCNN, MNN, with fp32 / fp16 / int8 precision
- Stereo camera calibration tools (`stereocalib/`)
- Motor control over GPIO with PWM (`motortest.py`)

## Project structure
```
src/
├── main_ultralytics_onnx.py   # Main entry point (Ultralytics YOLO + lane detection)
├── main_opencv_onnx.py        # Main entry point (OpenCV DNN + lane detection)
├── objdetect_ultralytics.py   # Object detection / distance logic (Ultralytics)
├── objdetect_yolov8n_opencv.py# Object detection (OpenCV DNN)
├── houghalgor.py              # Lane line detection
├── motortest.py               # Motor controller class
├── traffic.py                 # Traffic-sign handling
├── cam.py                     # Camera helper
├── stereocalib/               # Stereo calibration scripts
├── off/, .old/                # Older experiments
```

## Requirements
- Raspberry Pi with `picamera2` and two cameras (stereo setup)
- Motor driver connected via GPIO
- Python 3 with:
  ```bash
  pip install numpy opencv-python ultralytics torch gpiozero
  ```

## Models
Model weights are **not** included in this repository because of their size.
Place them in the following layout (or edit the paths in the main scripts):
```
model/trafficspeedsign/ultralytics/<modeltype>/v<version>/<format>/<precision>/trafficspeedsign_<precision>.<format>
```
Calibration files are expected at `assets/calib/stereo_maps.npz` and `assets/calib/line_maps.npz`
(generate them with the scripts in `src/stereocalib/`).

## Usage
```bash
cd src
python main_ultralytics_onnx.py --format onnx --precision fp16 --modeltype yolov8 --modelversion 1
```

| Argument | Default | Options |
|---|---|---|
| `--format` | `onnx` | `onnx`, `ncnn`, `mnn` |
| `--precision` | `fp16` | `fp32`, `fp16`, `int8` |
| `--modeltype` | `yolov8` | `yolov3`, `yolov5`, `yolov8`, `yolov11` |
| `--modelversion` | `1` | integer |

## Status
Completed. This project was developed as a graduation project and is no longer under activedevelopment. The code is shared here for reference and learning.

## License
Released under the MIT License.
