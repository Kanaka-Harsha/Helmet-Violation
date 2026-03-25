# Advanced Computer Vision & Deep Learning Toolkit

This repository contains multiple real-time computer vision and deep learning applications built using **OpenCV** and **Ultralytics YOLOv8**. The projects focus on object detection, tracking, safety compliance, and analytics across different scenarios.

## 🚀 Projects

### 1. Vehicle Classification & Analytics (`vehicle_classification/`)
A real-time traffic monitoring system that detects, classifies, and counts different types of vehicles as they cross a defined line.
- **Features:**
  - Classifies vehicles into distinct categories (Car, Bus, Truck, Motorbike, Ambulance, etc.).
  - Tracks vehicles using ByteTrack to prevent double-counting.
  - Interactive counting line adjustable in real-time via keyboard (`W`/`S` to move, `A`/`D` to rotate).
  - Modern transparent UI overlay for real-time statistics and FPS.
- **Usage:** Run `vehicle_classification.py`.

### 2. People Counting (`people_count/`)
An intelligent crowd analytics tool that monitors people entering and exiting a specific area or doorway.
- **Features:**
  - Tracks the 'person' class using YOLOv8 and ByteTrack matching.
  - Interactive setup: User clicks two points on the video feed to define the counting boundary line.
  - Directional tracking for distinct Entry and Exit counts.
- **Usage:** Run `people_count.py`.

### 3. Helmet Violation Detection (`helmet_violation/`)
A safety compliance monitoring system specifically designed to detect whether motorcycle riders are wearing helmets.
- **Features:**
  - Detects riders and classifies whether they are wearing a helmet or not.
  - Automatically captures and saves cropped evidence images of violations into a dedicated `no_helmet_bikes/` evidence folder.
  - Draws bounding boxes (Green for Helmet, Red for No Helmet) and confidence scores in real-time.
- **Usage:** Run `helmet_violation.py`.

### 4. Helper Utilities (`helper_codes/`)
Contains utility scripts used during project development such as splitting images/videos for dataset creation and testing transparent UI overlays.

## ⚙️ Requirements
- Python 3.x
- Ultralytics (`pip install ultralytics`)
- OpenCV (`pip install opencv-python`)
- Numpy, PyQt5 (for some UI features)

## 🔧 How to Run
Each main script is located in its respective folder and requires a video source. Before running, open the desired script and adjust the `VIDEO_PATH` variable to point to your input video (or set it to `0` for webcam).

```bash
cd vehicle_classification
python vehicle_classification.py
```
