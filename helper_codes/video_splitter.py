import cv2
import os
import random
import shutil

DATASET_DIR = "dataset"
TRAIN_RATIO = 0.8
VAL_RATIO = 0.1
TEST_RATIO=0.1
INTERVAL_SECONDS = 1

def create_directory_structure(base_dir):
    subdirs = [
        "train/images",
        "train/labels",
        "test/images",
        "test/labels",
        "val/images",
        "val/labels"
    ]
    
    if os.path.exists(base_dir):
        print(f"Directory {base_dir} already exists.")
    else:
        os.makedirs(base_dir)

    for subdir in subdirs:
        path = os.path.join(base_dir, subdir)
        os.makedirs(path, exist_ok=True)
        print(f"Created: {path}")

def split_video(video_path, video_name, dataset_dir, train_ratio=0.8, val_ratio=0.1, test_ratio=0.1, interval_seconds=1):
    """Splits video frames into train/val/test dataset folders."""
    
    if not os.path.exists(video_path):
        print(f"Error: Video file not found at {video_path}")
        return

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print("Error: Could not open video.")
        return

    frame_count = 0
    saved_count = 0
    
    fps = cap.get(cv2.CAP_PROP_FPS)
    if fps == 0:
        print("Error: Could not retrieve FPS.")
        return
        
    frame_interval = int(fps * interval_seconds)
    if frame_interval == 0:
        frame_interval = 1
        
    print(f"Processing video: {video_path} at {fps} FPS with interval {interval_seconds}s (every {frame_interval} frames)")
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
            
        if frame_count % frame_interval == 0:
            # Determine split
            rn = random.random()
            if rn < train_ratio:
                split = "train"
            elif rn < train_ratio + val_ratio:
                split = "val"
            else:
                split = "test"
                
            image_name = f"{video_name}_frame_{saved_count:06d}.jpg"
            save_path = os.path.join(dataset_dir, split, "images", image_name)
            
            cv2.imwrite(save_path, frame)
            
            label_name = f"{video_name}_frame_{saved_count:06d}.txt"
            label_path = os.path.join(dataset_dir, split, "labels", label_name)
            with open(label_path, 'w') as f:
                pass
                
            saved_count += 1
            if saved_count % 50 == 0:
                print(f"Saved {saved_count} images...")

        frame_count += 1

    cap.release()
    print(f"Done! Extracted {saved_count} images from {frame_count} frames.")
    print(f"Dataset location: {os.path.abspath(dataset_dir)}")

if __name__ == "__main__":
    create_directory_structure(DATASET_DIR)
    
    VIDEO_DIR = "Videos"
    if not os.path.exists(VIDEO_DIR):
        print(f"Error: Video directory '{VIDEO_DIR}' not found.")
        exit(1)

    video_extensions = ('.mp4', '.avi', '.mov', '.mkv')
    # Filter for the specific video file requested
    target_video = "Final CCTV Rec.mp4"
    video_files = [f for f in os.listdir(VIDEO_DIR) if f == target_video and f.lower().endswith(video_extensions)]

    if not video_files:
        print(f"No video files found in {VIDEO_DIR} matching '{target_video}'")
    else:
        print(f"Found {len(video_files)} videos: {video_files}")
        for video_file in video_files:
            video_path = os.path.join(VIDEO_DIR, video_file)
            video_name = os.path.splitext(video_file)[0]
            print(f"\n--- Processing {video_file} ---")
            split_video(video_path, video_name, DATASET_DIR, TRAIN_RATIO, VAL_RATIO, TEST_RATIO, INTERVAL_SECONDS)
