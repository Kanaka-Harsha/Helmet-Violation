import cv2
import numpy as np
from ultralytics import YOLO
from collections import defaultdict

# --- Configuration ---
MODEL_PATH = "best.pt" 
# Example: "rtsp://username:password@ip_address:554/stream"
VIDEO_SOURCE = "Videos/vlc-record-2026-02-18-12h37m29s-rtsp___192.168.1.10_Streaming_Channels_101--converted.mp4" 


TARGET_CLASSES = [0, 1, 2, 3, 4, 5, 6, 7] 
OUTPUT_FILE = "vehicle_counts.txt" 

# Interactive Line Configuration
# Initial line position (x1, y1, x2, y2)
# We'll start with a horizontal line in the middle
START_LINE = [100, 500, 1800, 500] 

# Visuals
LINE_COLOR = (0, 0, 255) # Red
TEXT_COLOR = (255, 255, 255)
line_thickness = 3

def write_counts_to_file(counts):
    """Writes the current vehicle counts to a text file."""
    try:
        with open(OUTPUT_FILE, "w") as f:
            f.write("Vehicle Counts:\n")
            f.write("===============\n")
            for cls_name, count in counts.items():
                f.write(f"{cls_name}: {count}\n")
    except Exception as e:
        print(f"Error writing to file: {e}")

def main():
    # 1. Load Model
    try:
        model = YOLO(MODEL_PATH)
        print(f"✅ Model loaded from {MODEL_PATH}")
    except Exception as e:
        print(f"❌ Error loading model: {e}")
        print("Make sure 'best.pt' is in the same directory or update MODEL_PATH.")
        return

    # 2. Open Video Source
    cap = cv2.VideoCapture(VIDEO_SOURCE)
    if not cap.isOpened():
        print(f"❌ Error: Could not open video source {VIDEO_SOURCE}")
        return

    # Get video info
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    
    # Resize window for better view
    cv2.namedWindow("Vehicle Counting", cv2.WINDOW_NORMAL)
    cv2.resizeWindow("Vehicle Counting", 1280, 720)

    # State variables
    line_pos = list(START_LINE) # [x1, y1, x2, y2]
    previous_positions = {} # track_id -> (x, y)
    counts = defaultdict(int)
    counted_ids = set()

    print("🚀 Tracking started.")
    print("Controls:")
    print("  W/S: Move Line Up/Down")
    print("  A/D: Move Line Left/Right")
    print("  Q: Quit")

    while True:
        ret, frame = cap.read()
        if not ret:
            print("End of stream.")
            break

        # 3. Interactive Line Adjustment (WASD)
        # Check key presses to move the line
        key = cv2.waitKey(1) & 0xFF
        step = 10 # Pixels to move per key press

        if key == ord('q'):
            break
        elif key == ord('w'): # Up
            line_pos[1] -= step
            line_pos[3] -= step
        elif key == ord('s'): # Down
            line_pos[1] += step
            line_pos[3] += step
        elif key == ord('a'): # Left
            line_pos[0] -= step
            line_pos[2] -= step
        elif key == ord('d'): # Right
            line_pos[0] += step
            line_pos[2] += step

        # Clamp line to frame boundaries
        line_pos[0] = max(0, min(width, line_pos[0]))
        line_pos[2] = max(0, min(width, line_pos[2]))
        line_pos[1] = max(0, min(height, line_pos[1]))
        line_pos[3] = max(0, min(height, line_pos[3]))

        # Define line start/end points for calculations
        line_p1 = (line_pos[0], line_pos[1])
        line_p2 = (line_pos[2], line_pos[3])

        # 4. Run Tracking
        # persist=True is crucial for ID tracking across frames
        results = model.track(frame, persist=True, verbose=False, classes=TARGET_CLASSES)
        
        # 5. Process Tracks
        if results[0].boxes.id is not None:
            boxes = results[0].boxes.xyxy.cpu()
            track_ids = results[0].boxes.id.int().cpu().tolist()
            cls_ids = results[0].boxes.cls.int().cpu().tolist()

            for box, track_id, cls in zip(boxes, track_ids, cls_ids):
                x1, y1, x2, y2 = box.tolist()
                cx, cy = int((x1 + x2) / 2), int((y1 + y2) / 2)

                # Draw bounding box (optional, usually handled by model.plot() but good for custom view)
                # cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)
                # cv2.putText(frame, f"ID: {track_id}", (int(x1), int(y1)-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

                # Check for Crossing
                if track_id in previous_positions:
                    prev_cx, prev_cy = previous_positions[track_id]
                    
                    # Vector math to check intersection
                    # We check if the segment (prev_center -> curr_center) intersects the counting line
                    if track_id not in counted_ids:
                        if intersect((prev_cx, prev_cy), (cx, cy), line_p1, line_p2):
                            counts[model.names[cls]] += 1
                            counted_ids.add(track_id)
                            write_counts_to_file(counts)
                            # Draw a hit effect
                            cv2.line(frame, line_p1, line_p2, (0, 255, 0), 5) 

                # Update previous position
                previous_positions[track_id] = (cx, cy)

        # 6. Draw Visuals
        # Draw the tracking results
        annotated_frame = results[0].plot()

        # Draw the interactive line
        cv2.line(annotated_frame, line_p1, line_p2, LINE_COLOR, line_thickness)
        
        # Display Controls and Counts
        y_offset = 30
        cv2.putText(annotated_frame, "Controls: WASD to move line", (10, y_offset), cv2.FONT_HERSHEY_PLAIN, 1.5, TEXT_COLOR, 2)
        
        y_offset += 30
        cv2.putText(annotated_frame, "Counts:", (10, y_offset), cv2.FONT_HERSHEY_PLAIN, 1.5, TEXT_COLOR, 2)
        
        for cls_name, count in counts.items():
            y_offset += 30
            cv2.putText(annotated_frame, f"  {cls_name}: {count}", (10, y_offset), cv2.FONT_HERSHEY_PLAIN, 1.5, (0, 255, 255), 2)

        cv2.imshow("Vehicle Counting", annotated_frame)

    cap.release()
    cv2.destroyAllWindows()

def start_cw(p1, p2, p3):
    return (p2[0] - p1[0]) * (p3[1] - p1[1]) - (p2[1] - p1[1]) * (p3[0] - p1[0])

def intersect(p1, p2, p3, p4):
    """
    Returns True if line segment p1-p2 intersects line segment p3-p4
    """
    # Check if p1 and p2 are on opposite sides of line p3-p4
    # AND if p3 and p4 are on opposite sides of line p1-p2
    val1 = start_cw(p3, p4, p1)
    val2 = start_cw(p3, p4, p2)
    val3 = start_cw(p1, p2, p3)
    val4 = start_cw(p1, p2, p4)

    return ((val1 > 0 and val2 < 0) or (val1 < 0 and val2 > 0)) and \
           ((val3 > 0 and val4 < 0) or (val3 < 0 and val4 > 0))

if __name__ == "__main__":
    main()
