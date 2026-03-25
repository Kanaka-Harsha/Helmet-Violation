import cv2
import os
import math
import numpy as np
from ultralytics import YOLO

os.environ["OPENCV_FFMPEG_CAPTURE_OPTIONS"] = "loglevel;quiet"
os.environ["OPENCV_LOG_LEVEL"] = "QUIET"

VIDEO_PATH = "Videos/vehicle_and_helmet.mp4" # Replace with your video file
HELMET_MODEL_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "best.pt")

OUTPUT_FOLDER = "violations_evidence"
NO_HELMET_FOLDER = "no_helmet_bikes"

# Helmet Model (best.pt) Classes
HELMET_CLASS_IDS = {
    'helmet': 0, # 'wiht_helmet'
    'no_helmet': 1 # 'without_helmet'
}

# 1. Create Output Folder
if not os.path.exists(OUTPUT_FOLDER):
    os.makedirs(OUTPUT_FOLDER)
    print(f"📁 Created folder: {OUTPUT_FOLDER}")

if not os.path.exists(NO_HELMET_FOLDER):
    os.makedirs(NO_HELMET_FOLDER)
    print(f"📁 Created folder: {NO_HELMET_FOLDER}")

def main():
    print("🧠 Loading Helmet Model...")
    helmet_model = YOLO(HELMET_MODEL_PATH)
    
    print(f"🎥 Opening video file: {VIDEO_PATH}")
    cap = cv2.VideoCapture(VIDEO_PATH)

    if not cap.isOpened():
        print(f"❌ Error: Could not open video file {VIDEO_PATH}")
        return

    # Resize frame for processing to speed up inference
    disp_width = 1024
    
    frame_count = 0
    
    cv2.namedWindow("Helmet Detection", cv2.WINDOW_NORMAL)
    
    print("🎥 Starting Video Processing...")
    
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            print("End of video stream or cannot fetch the frame.")
            break
            
        disp_height = int(frame.shape[0] * (disp_width / frame.shape[1]))
        frame = cv2.resize(frame, (disp_width, disp_height))
        
        frame_count += 1
        
        # Run inference on helmet model
        helmet_results = helmet_model(frame, imgsz=480, verbose=False)[0]
        
        canvas = frame.copy()
        
        # Extract and Draw from Helmet Model directly
        for box in helmet_results.boxes.data.tolist():
            x1, y1, x2, y2, conf, cls_id = box
            x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
            cls_id = int(cls_id)
            
            if cls_id == HELMET_CLASS_IDS['no_helmet']:
                # Draw Red Box for No Helmet
                cv2.rectangle(canvas, (x1, y1), (x2, y2), (0, 0, 255), 2)
                cv2.putText(canvas, f"No Helmet {conf:.2f}", (x1, max(10, y1 - 10)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
                
                # Save crop
                img_name = f"{NO_HELMET_FOLDER}/no_helmet_{frame_count}_{x1}_{y1}.jpg"
                h, w, _ = frame.shape
                bx1, by1 = max(0, x1), max(0, y1)
                bx2, by2 = min(w, x2), min(h, y2)
                crop_img = frame[by1:by2, bx1:bx2]
                
                try:
                    cv2.imwrite(img_name, crop_img)
                except Exception as e:
                    pass

            elif cls_id == HELMET_CLASS_IDS['helmet']:
                # Draw Green Box for Helmet
                cv2.rectangle(canvas, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.putText(canvas, f"Helmet {conf:.2f}", (x1, max(10, y1 - 10)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
                
        cv2.putText(canvas, "Press 'q' to quit.", (10, disp_height - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
        
        cv2.imshow("Helmet Detection", canvas)
        
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == '__main__':
    main()