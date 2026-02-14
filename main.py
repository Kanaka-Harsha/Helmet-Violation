import cv2
import os
import math
from ultralytics import YOLO

VIDEO_PATH = "Videos/Main.mp4"
MODEL_PATH = "best.pt"
OUTPUT_FOLDER = "violations_evidence"


CLASS_IDS = {
    'bike': 0,         # Replace with your actual ID for bike
    'no_helmet': 1,    # Replace with your actual ID for no_helmet
    'helmet': 2,       # Replace with your actual ID for helmet
    'numberplate': 3   # Replace with your actual ID for numberplate
}


def get_center(box):
    """Calculates the center point (x, y) of a bounding box."""
    x1, y1, x2, y2 = box
    return int((x1 + x2) / 2), int((y1 + y2) / 2)

def calculate_distance(p1, p2):
    """Calculates Euclidean distance between two points."""
    return math.sqrt((p1[0] - p2[0])**2 + (p1[1] - p2[1])**2)

def is_overlap(box1, box2):
    """
    Simple check if box1 (e.g., person/head) is somewhat inside box2 (bike).
    We check if the center of box1 is inside box2.
    """
    c_x, c_y = get_center(box1)
    x1, y1, x2, y2 = box2
    return x1 < c_x < x2 and y1 < c_y < y2


# 1. Create Output Folder
if not os.path.exists(OUTPUT_FOLDER):
    os.makedirs(OUTPUT_FOLDER)
    print(f"📁 Created folder: {OUTPUT_FOLDER}")

# 2. Load the Model
print("🧠 Loading Model...")
model = YOLO(MODEL_PATH)

# 3. Open Video
cap = cv2.VideoCapture(VIDEO_PATH)
if not cap.isOpened():
    print("❌ Error: Could not open video.")
    exit()

frame_count = 0

print("🎥 Starting Surveillance...")

while True:
    ret, frame = cap.read()
    if not ret:
        break 
    
    frame_count += 1
    
    # Run Inference
    results = model(frame, verbose=False)[0] 
    
    # Sort detections into lists by class for easier processing
    bikes = []
    no_helmets = []
    plates = []
    
    # Loop through all detections in this frame
    for box in results.boxes.data.tolist():
        x1, y1, x2, y2, conf, cls_id = box
        x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
        cls_id = int(cls_id)
        
        if cls_id == CLASS_IDS['bike']:
            bikes.append([x1, y1, x2, y2])
        elif cls_id == CLASS_IDS['no_helmet']:
            no_helmets.append([x1, y1, x2, y2])
        elif cls_id == CLASS_IDS['numberplate']:
            plates.append([x1, y1, x2, y2])

    # 🕵️ LOGIC: Associate detections
    for bike_box in bikes:
        bike_center = get_center(bike_box)
        
        # Check 1: Does this bike have a "No Helmet" detected on/near it?
        violation_detected = False
        for nh_box in no_helmets:
            # We check if the 'no helmet' center is inside the 'bike' box
            if is_overlap(nh_box, bike_box):
                violation_detected = True
                break # Found the violation for this bike
        
        # Check 2: If violation, find the closest number plate
        if violation_detected:
            closest_plate = None
            min_dist = 99999
            
            for plate_box in plates:
                plate_center = get_center(plate_box)
                dist = calculate_distance(bike_center, plate_center)
                
                # Heuristic: Plate must be close to bike center (adjust threshold as needed)
                if dist < min_dist and is_overlap(plate_box, bike_box): 
                    min_dist = dist
                    closest_plate = plate_box
            
            # 📸 Check 3: Capture if we have both Bike + Violation + Plate
            if closest_plate:
                # Format filenames
                img_name_bike = f"{OUTPUT_FOLDER}/frame_{frame_count}_bike.jpg"
                img_name_plate = f"{OUTPUT_FOLDER}/frame_{frame_count}_plate.jpg"
                
                # Crop and Save Bike
                bx1, by1, bx2, by2 = bike_box
                # Clamp coordinates to frame dimensions to avoid errors
                h, w, _ = frame.shape
                bx1, by1 = max(0, bx1), max(0, by1)
                bx2, by2 = min(w, bx2), min(h, by2)
                
                crop_bike = frame[by1:by2, bx1:bx2]
                cv2.imwrite(img_name_bike, crop_bike)
                
                # Crop and Save Plate
                px1, py1, px2, py2 = closest_plate
                px1, py1 = max(0, px1), max(0, py1)
                px2, py2 = min(w, px2), min(h, py2)
                
                crop_plate = frame[py1:py2, px1:px2]
                cv2.imwrite(img_name_plate, crop_plate)
                
                print(f"📸 VIOLATION CAPTURED: Frame {frame_count}")
                
                # Draw visual feedback on the live feed (Optional)
                cv2.rectangle(frame, (bx1, by1), (bx2, by2), (0, 0, 255), 2) # Red box on bike
                cv2.rectangle(frame, (px1, py1), (px2, py2), (0, 255, 255), 2) # Yellow on plate

    # Show the video running
    cv2.imshow('Traffic Surveillance', frame)
    
    # Press 'q' to quit
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
print("✅ Processing Complete.")