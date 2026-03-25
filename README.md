# SMART TRAFFIC MONITORING

![Python](https://img.shields.io/badge/Python-3.x-blue.svg)
![OpenCV](https://img.shields.io/badge/OpenCV-4.x-green.svg)
![YOLOv8](https://img.shields.io/badge/YOLOv8-Ultralytics-yellow.svg)

Welcome to the **Deep Learning and Computer Vision Projects** repository! This collection features three robust, real-time computer vision applications powered by **Ultralytics YOLOv8** and **OpenCV**. These tools are engineered to autonomously track, analyze, and enforce safety compliances in dynamic real-world environments.

---

## 🚀 Projects Overview

### 1. 🚦 Vehicle Classification & Analytics (`vehicle_classification/`)
A high-performance traffic monitoring system designed to classify, track, and count vehicles accurately on busy roads.

- **How it Works**: Uses a custom-trained YOLO model (`best.pt`) to detect a variety of vehicle types (Car, Bus, Truck, Motorbike, Ambulance, Auto Rickshaw, etc.). It leverages the **ByteTrack** algorithm to maintain identity tracking across frames to ensure absolute accuracy when vehicles cross a user-defined threshold.
- **Key Features**:
  - ✨ Multi-class vehicle categorization.
  - 🎮 **Interactive UI**: Use `W`/`S` (Up/Down) to translate the counting line and `A`/`D` (Left/Right) to rotate the counting line dynamically while the stream is running!
  - 📊 Real-time dynamic overlay dashboard displaying Frame Per Second (FPS) and itemized classification counts.

### 2. 👥 People Counting with Interactive Boundaries (`people_count/`)
An intelligent crowd analytics tool that monitors human foot traffic precisely in entryways, corridors, or storefronts.

- **How it Works**: Uses the YOLOv8 person-detection models (`yolov8npep.pt`, `yolov8s.pt`) integrated with ByteTrack. It mathematically evaluates when a tracked individual fully crosses a geometric plane.
- **Key Features**:
  - 🖱️ **Point-and-Click Setup**: Setup requires zero hardcoded coordinates! The user simply clicks two points on the video feed to define their custom entry/exit counting boundary.
  - ↔️ **Directional Logic**: Intelligently differentiates between "Entry" and "Exit" events based on trajectory crossing.
  - 🎨 Transparent heads-up display showcasing live entry and exit metrics.

### 3. 🪖 Helmet Violation Detection (`helmet_violation/`)
A critical automated safety compliance tool intended to enforce traffic laws and workplace safety protocols.

- **How it Works**: Employs a highly specialized YOLO model (`best.pt`) explicitly trained to detect motorcycle riders and infer the presence (or absence) of a helmet. 
- **Key Features**:
  - 🛑 **Violation Archiving**: The system automatically captures and saves cropped evidence images (`.jpg`) of riders detected *without* a helmet directly to a dedicated `no_helmet_bikes/` folder. This provides an immediate audit trail!
  - 🟩 **Visual Annotations**: Draws intuitive bounding boxes (Green for Helmet ✅, Red for No Helmet ❌) and logs confidence percentage values.

### 4. 🛠️ Helper Utilities (`helper_codes/`)
Supporting Python utilities designed to accelerate dataset preparation and UI feature-testing:
- `video_splitter.py` / `image_splitter.py`: Helper scripts to segment bulk video/image data.
- Interface test scripts designed to prototype the transparent HUD elements integrated heavily into the primary modules.

---

## ⚙️ Installation & Requirements

Ensure you have Python 3.8+ installed on your system. 

**1. Clone the repository**
```bash
git clone https://github.com/Kanaka-Harsha/Helmet-Violation.git
cd Helmet-Violation
```

**2. Install dependencies**
It is highly recommended to install dependencies within a virtual environment.
```bash
pip install ultralytics
pip install opencv-python
pip install numpy
pip install PyQt5
pip install PyGetWindow
pip install mss
```

*(Note: Depending on your hardware, we recommend installing the PyTorch build tailored for your CUDA architecture to utilize GPU acceleration for maximum FPS!)*

---

## 🏃 Getting Started

Each project module runs completely autonomously. Simply navigate to the folder and execute the main Python script.

### Running Vehicle Analytics
```bash
cd vehicle_classification
# Update VIDEO_PATH in the script if you want to use a custom video
python vehicle_classification.py
```

### Running People Counting
```bash
cd people_count
# Follow the on-screen prompt: Click 2 points to draw the line, press 'C' to confirm tracking!
python people_count.py
```

### Running Helmet Violation
```bash
cd helmet_violation
python helmet_violation.py
```

*💡 Pro Tip: For any of the above scripts, you can open the `.py` file and change the `VIDEO_PATH` variable to `0` to test real-time inference on your connected webcam instead of a static video file!*
