# overexposure.py
import cv2
import os
import logging
from snap_sort.utils.file_manager import FileManager
from snap_sort.utils.image_loader import ImageLoader
from concurrent.futures import ThreadPoolExecutor

def is_tone_level(image, tone_level):
    """Check if the image's tone level matches the specified tone_level ('low', 'mid', or 'high')."""
    image = ImageLoader.resize_image(image)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    mean_brightness = gray.mean()  # Calculate the mean brightness

    # Define brightness thresholds for low, medium, and high tones
    if tone_level == 'low':
        return mean_brightness < 85  # Low tone (bottom third)
    elif tone_level == 'mid':
        return 85 <= mean_brightness < 170  # Medium tone (middle third)
    elif tone_level == 'high':
        return mean_brightness >= 170  # High tone (top third)
    return False

def process_image(filename, folder_path, target_folder, tone_level):
    """Process a single image: check if it matches the specified tone level and move if necessary."""
    image_path = os.path.join(folder_path, filename)
    logging.info(f"Classifying image: {image_path}")
    image = cv2.imread(image_path)
    if image is not None:
        if is_tone_level(image, tone_level):
            FileManager.move_file(image_path, target_folder)
    else:
        logging.warning(f"Failed to read image: {image_path}")

def classify_images_by_tone(folder_path, tone_level):
    """Classify images in the given folder based on a specified tone level and move matching images."""
    image_files = [f for f in os.listdir(folder_path) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]

    # Create a folder for the specified tone level
    target_folder = os.path.join(folder_path, f"{tone_level}_tone")
    os.makedirs(target_folder, exist_ok=True)

    # Determine the number of threads to use
    num_threads = min(32, os.cpu_count() + 4)

    with ThreadPoolExecutor(max_workers=num_threads) as executor:
        futures = []
        for filename in image_files:
            futures.append(executor.submit(process_image, filename, folder_path, target_folder, tone_level))

        # Optionally, wait for all futures to complete and handle exceptions
        for future in futures:
            try:
                future.result()
            except Exception as e:
                logging.error(f"Error processing image: {e}")

    # Update the redo file
    FileManager.update_redo_file(folder_path, target_folder)
