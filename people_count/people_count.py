import cv2
import sys
import numpy as np
from ultralytics import YOLO
import mss
import pygetwindow as gw
from PyQt5 import QtWidgets, QtGui, QtCore
import threading

import os
os.environ["OPENCV_FFMPEG_CAPTURE_OPTIONS"] = "loglevel;quiet"
os.environ["OPENCV_LOG_LEVEL"] = "QUIET"

# Configuration
# VIDEO_PATH = "Videos/peoplecount.mp4" # Replace with your video path or 0 for webcam
VIDEO_PATH = "Videos/peoplecount.mp4"
MODEL_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "yolov8npep.pt")
TARGET_CLASSES = [0] # List of class indices to track (e.g., 0 for 'person'). Adjust based on your model's classes.

# Globals for adjustable line
line_points = []
drawing = False
line_defined = False

def draw_line(event, x, y, flags, param):
    """
    Mouse callback function to draw an adjustable line on the first frame.
    Allows user to click two points visually to set the counting boundary.
    """
    global line_points, drawing, line_defined
    
    # Left click starts drawing
    if event == cv2.EVENT_LBUTTONDOWN:
        if not line_defined:
            if len(line_points) == 0:
                line_points = [(x, y)]
            elif len(line_points) == 1:
                line_points.append((x, y))
                line_defined = True
                
def get_side(px, py, line_p1, line_p2):
    x1, y1 = line_p1
    x2, y2 = line_p2
    value = (x2 - x1) * (py - y1) - (y2 - y1) * (px - x1)

    # tolerance zone (important for crowd stability)
    if abs(value) < 20:
        return 0

    return 1 if value > 0 else -1

def main():
    global line_points, line_defined

    print(f"🎥 Opening video file: {VIDEO_PATH}")
    cap = cv2.VideoCapture(VIDEO_PATH)

    if not cap.isOpened():
        print("❌ Error: Could not open video file.")
        return

    # Read the first frame to allow the user to draw the line
    ret, first_frame = cap.read()
    if not ret:
        print("❌ Error: Cannot read the first frame from the video.")
        return
        
    # Resize frame for viewing to fit on screen
    disp_width = 1024
    disp_height = int(first_frame.shape[0] * (1024 / first_frame.shape[1]))
    first_frame = cv2.resize(first_frame, (disp_width, disp_height))

    print("✏️  ADJUSTABLE LINE SETUP:")
    print("Please click TWO points on the opened video window to define your counting line.")
    print("Press 'c' when you are done to start tracking.")
    print("Press 'r' to reset and redraw the line.")
    
    window_name = "Draw Line (Click 2 points, 'c'=confirm, 'r'=redraw)"
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(window_name, disp_width, disp_height)
    cv2.setMouseCallback(window_name, draw_line)

    while True:
        img_copy = first_frame.copy()
        
        cv2.putText(img_copy, "Click 2 points for line.", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        cv2.putText(img_copy, "Press 'c' to confirm, 'r' to redraw.", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        
        if len(line_points) == 1:
            cv2.circle(img_copy, line_points[0], 5, (0, 0, 255), -1)
        elif len(line_points) == 2:
            cv2.line(img_copy, line_points[0], line_points[1], (0, 255, 0), 2)
            cv2.circle(img_copy, line_points[0], 5, (0,0,255), -1)
            cv2.circle(img_copy, line_points[1], 5, (0,0,255), -1)
            
        cv2.imshow(window_name, img_copy)
            
        key = cv2.waitKey(1) & 0xFF
        if key == ord('c') and line_defined:
            break
        elif key == ord('r'):
            line_defined = False
            line_points = []
            
    cv2.destroyWindow(window_name)
    print(f"✅ Line defined from {line_points[0]} to {line_points[1]}")
    
    print("🧠 Loading YOLOv8 Model...")
    model = YOLO(MODEL_PATH)

    track_history = {}
    entry_count = 0
    exit_count = 0
    previous_side = {}
    counted_ids = {}

    cv2.namedWindow("People Counting", cv2.WINDOW_NORMAL)
    
    cap.set(cv2.CAP_PROP_POS_FRAMES, 0) # reset frame back to 0

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            print("End of video stream or cannot fetch the frame.")
            break

        frame = cv2.resize(frame, (disp_width, disp_height))

        # Using imgsz=480 for a speed up and ByteTrack tracker
        results = model.track(
            frame, 
            persist=True, 
            tracker="bytetrack.yaml", 
            conf=0.2, 
            iou=0.5, 
            classes=TARGET_CLASSES, 
            verbose=False, 
            imgsz=480
        )
        
        tracking_data = []
        if results[0].boxes and results[0].boxes.id is not None:
            boxes = results[0].boxes.xyxy.cpu().numpy()
            track_ids = results[0].boxes.id.int().cpu().numpy()
            confs = results[0].boxes.conf.cpu().numpy()
            for box, track_id, conf in zip(boxes, track_ids, confs):
                x1, y1, x2, y2 = map(int, box)
                tracking_data.append(((x1, y1, x2, y2), int(track_id), float(conf)))

        canvas = frame.copy()

        cv2.line(canvas, line_points[0], line_points[1], (0, 255, 255), 3)
        cv2.putText(canvas, "Counting Line", (line_points[0][0], line_points[0][1]-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)

        for (x1, y1, x2, y2), track_id, conf in tracking_data:
            cx = int((x1 + x2) / 2)
            cy = int((y1 + y2) / 2)
            current_point = (cx, cy)
            
            if track_id not in track_history:
                track_history[track_id] = []
            track_history[track_id].append(current_point)
            
            if len(track_history[track_id]) > 30:
                track_history[track_id].pop(0)
                
            current_side = get_side(cx, cy, line_points[0], line_points[1])

            if track_id not in previous_side:
                previous_side[track_id] = current_side
                counted_ids[track_id] = False

            prev_side = previous_side[track_id]

            if not counted_ids[track_id]:
                if prev_side != 0 and current_side != 0:
                    # ENTRY
                    if prev_side < 0 and current_side > 0:
                        entry_count += 1
                        counted_ids[track_id] = True
                        cv2.line(canvas, line_points[0], line_points[1], (0, 255, 0), 6)
                        print(f"👤 Person Entered! Total Entry: {entry_count} Tracking ID: {track_id}")
                    # EXIT
                    elif prev_side > 0 and current_side < 0:
                        exit_count += 1
                        counted_ids[track_id] = True
                        cv2.line(canvas, line_points[0], line_points[1], (0, 0, 255), 6)
                        print(f"👤 Person Exited! Total Exit: {exit_count} Tracking ID: {track_id}")

            previous_side[track_id] = current_side

            cv2.rectangle(canvas, (x1, y1), (x2, y2), (255, 0, 0), 2)
            cv2.circle(canvas, current_point, 4, (0, 0, 255), -1)
            cv2.putText(canvas, f"ID: {track_id} C:{conf:.2f}", (x1, max(10, y1 - 10)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)

        # Transparent overlay for stats box
        overlay_stats = canvas.copy()
        cv2.rectangle(overlay_stats, (10, 10), (300, 110), (0, 0, 0), -1)
        cv2.addWeighted(overlay_stats, 0.6, canvas, 0.4, 0, canvas)
        
        cv2.putText(canvas, f"Entry: {entry_count}", (20, 50), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 255, 0), 3)
        cv2.putText(canvas, f"Exit: {exit_count}", (20, 90), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 0, 255), 3)
        cv2.putText(canvas, "Press 'q' to quit.", (10, disp_height - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
        
        cv2.imshow("People Counting", canvas)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
