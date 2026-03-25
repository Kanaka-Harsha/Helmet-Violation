import cv2
import time
import sys
import math
import numpy as np
from collections import defaultdict
from ultralytics import YOLO

import os
os.environ["OPENCV_FFMPEG_CAPTURE_OPTIONS"] = "loglevel;quiet"
os.environ["OPENCV_LOG_LEVEL"] = "QUIET"

# ============================
# CONFIGURATION
# ============================

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
VIDEO_PATH = os.path.join(SCRIPT_DIR, "..", "videos", "Gates\Video_20260228131918\Gate NVR-1_Gate 2 Main Road 1_20260224092741_20260224093035.mp4")
MODEL_PATH = os.path.join(SCRIPT_DIR, "best.pt")

CONF_THRESHOLD = 0.20
TRACK_HISTORY_LENGTH = 20

WINDOW_NAME = "Advanced Vehicle Analytics"

CLASS_NAMES = {
    0: "ambulance",
    2: "auto rickshaw",
    3: "bicycle",
    4: "bus",
    5: "car",
    10: "motorbike",
    14: "scooter",
    16: "taxi",
    18: "truck",
    19: "van"
}

TARGET_CLASSES = list(CLASS_NAMES.keys())

# ============================
# GEOMETRY UTILITIES
# ============================

def ccw(A, B, C):
    return (C[1]-A[1])*(B[0]-A[0]) > (B[1]-A[1])*(C[0]-A[0])

def intersect(A, B, C, D):
    return ccw(A,C,D) != ccw(B,C,D) and ccw(A,B,C) != ccw(A,B,D)

# ============================
# LINE CALCULATION
# ============================

def compute_line(cx, cy, length, angle):

    dx = (length/2) * math.cos(angle)
    dy = (length/2) * math.sin(angle)

    pt1 = (int(cx - dx), int(cy - dy))
    pt2 = (int(cx + dx), int(cy + dy))

    return pt1, pt2

# ============================
# MAIN SYSTEM
# ============================

def main():
    print("🧠 Loading AI Model...")
    model = YOLO(MODEL_PATH)
    
    print(f"🎥 Opening video file: {VIDEO_PATH}")
    cap = cv2.VideoCapture(VIDEO_PATH)

    if not cap.isOpened():
        print("❌ Error: Could not open video file.")
        return
        
    # Set a fixed processing resolution to improve FPS
    width = 1024
    height = 576
    
    line_cx = width // 2
    line_cy = height // 2
    line_length = width
    line_angle = 0.0
    
    move_step = 20
    angle_step = np.deg2rad(5)
    
    track_history = defaultdict(list)
    total_counts = {cls: set() for cls in TARGET_CLASSES}
    
    prev_time = time.time()
    
    print("🎥 Starting Video Processing...")
    
    cv2.namedWindow(WINDOW_NAME, cv2.WINDOW_NORMAL)
    
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            print("End of video stream or cannot fetch the frame.")
            break
            
        # Resize frame to speed up processing
        frame = cv2.resize(frame, (width, height))
            
        current_time = time.time()
        fps = 1 / (current_time - prev_time) if current_time - prev_time > 0 else 0
        prev_time = current_time

        pt1, pt2 = compute_line(line_cx, line_cy, line_length, line_angle)

        results = model.track(
            frame,
            classes=TARGET_CLASSES,
            persist=True,
            tracker="bytetrack.yaml",
            verbose=False
        )

        canvas = frame.copy()

        if results and results[0].boxes.id is not None:
            boxes = results[0].boxes.xyxy.cpu()
            ids = results[0].boxes.id.int().cpu().tolist()
            cls_ids = results[0].boxes.cls.int().cpu().tolist()
            confs = results[0].boxes.conf.cpu().tolist()

            for box, track_id, cls_id, conf in zip(boxes, ids, cls_ids, confs):
                if conf < CONF_THRESHOLD:
                    continue

                x1, y1, x2, y2 = map(int, box)
                cx = int((x1+x2)/2)
                cy = int((y1+y2)/2)

                track = track_history[track_id]

                if len(track) > 0:
                    prev_pt = track[-1]
                    if track_id not in total_counts[cls_id]:
                        if intersect(prev_pt, (cx,cy), pt1, pt2):
                            total_counts[cls_id].add(track_id)
                            print(f"✅ {CLASS_NAMES[cls_id].upper()} #{track_id} crossed line")

                track.append((cx,cy))
                if len(track) > TRACK_HISTORY_LENGTH:
                    track.pop(0)

                color = (0, 255, 0)
                cv2.rectangle(canvas, (x1, y1), (x2, y2), color, 2)

                label = f"{CLASS_NAMES[cls_id]} #{track_id} {conf:.2f}"
                cv2.putText(canvas, label, (x1, y1-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

                if len(track) > 1:
                    pts = np.array(track, np.int32).reshape((-1, 1, 2))
                    cv2.polylines(canvas, [pts], False, (0, 255, 255), 2)

        # Draw Line
        cv2.line(canvas, pt1, pt2, (0, 0, 255), 3)

        # Overlay Stats
        active_counts = {k: v for k, v in total_counts.items() if len(v) > 0}
        panel_height = max(40, 65 + 25 * len(active_counts))
        
        # Optimized transparent overlay
        sub_img = canvas[10:panel_height, 10:350]
        black_rect = np.zeros(sub_img.shape, dtype=np.uint8)
        canvas[10:panel_height, 10:350] = cv2.addWeighted(sub_img, 0.6, black_rect, 0.4, 1.0)
        
        cv2.putText(canvas, f"FPS: {fps:.1f}", (20, 35), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
        cv2.putText(canvas, "--- LINE CROSSINGS ---", (20, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
        
        cv2.putText(canvas, "Press 'q' to quit. W/S to move UP/DOWN. A/D to rotate.", (10, height - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
        
        y_offset = 85
        for cls_id, ids in active_counts.items():
            count = len(ids)
            text = f"{CLASS_NAMES[cls_id].upper()}: {count}"
            cv2.putText(canvas, text, (20, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
            y_offset += 25

        cv2.imshow(WINDOW_NAME, canvas)

        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break
        elif key == ord('w'):
            line_cy -= move_step
        elif key == ord('s'):
            line_cy += move_step
        elif key == ord('a'):
            line_angle -= angle_step
        elif key == ord('d'):
            line_angle += angle_step
        elif key == ord('z'):
            line_cx -= move_step
        elif key == ord('x'):
            line_cx += move_step

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
