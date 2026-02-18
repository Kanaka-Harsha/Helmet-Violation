import os
import shutil
import math
import random

# Configuration
SOURCE_DIR = "dataset/images"
OUTPUT_DIR = "labeling_batches"
NUM_FOLDERS = 30

def distribute_images():
    # 1. Collect all image files
    all_images = []
    print("Collecting images...")
    for root, dirs, files in os.walk(SOURCE_DIR):
        for file in files:
            if file.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp')):
                all_images.append(os.path.join(root, file))
    
    total_images = len(all_images)
    if total_images == 0:
        print("No images found!")
        return

    print(f"Found {total_images} images.")

    # 2. Shuffle images
    random.shuffle(all_images)

    # 3. Calculate split size
    images_per_folder = math.ceil(total_images / NUM_FOLDERS)
    print(f"Distributing roughly {images_per_folder} images per folder.")

    # 4. Create folders and distribute
    if not os.path.exists(OUTPUT_DIR):
        os.makedirs(OUTPUT_DIR)

    for i in range(NUM_FOLDERS):
        folder_name = f"batch_{i+1}"
        folder_path = os.path.join(OUTPUT_DIR, folder_name)
        
        if not os.path.exists(folder_path):
            os.makedirs(folder_path)
        
        # Determine slice for this batch
        start_idx = i * images_per_folder
        end_idx = min((i + 1) * images_per_folder, total_images)
        batch_images = all_images[start_idx:end_idx]
        
        print(f"  - Copying {len(batch_images)} images to {folder_name}...")
        
        for img_path in batch_images:
            try:
                shutil.copy(img_path, folder_path)
            except Exception as e:
                print(f"Error copying {img_path}: {e}")

    print("\n✅ Distribution Complete!")
    print(f"Images are ready in '{OUTPUT_DIR}' folder.")

if __name__ == "__main__":
    distribute_images()
