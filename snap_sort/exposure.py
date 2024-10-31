# overexposure.py
import cv2
import os
import logging
from snap_sort.utils.file_manager import FileManager
from snap_sort.utils.image_loader import ImageLoader
from concurrent.futures import ThreadPoolExecutor

def is_overexposed(image):
    """Check if the image is overexposed by analyzing a central region of interest (ROI)."""
    image = ImageLoader.resize_image(image)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    hist = cv2.calcHist([gray], [0], None, [256], [0, 256])

    return hist[-1] > 0.1 * image.size

def process_image(filename, folder_path, overexposed_folder):
    """Process a single image: check if overexposed and move if necessary."""
    image_path = os.path.join(folder_path, filename)
    logging.info(f"Classifying image: {image_path}")
    image = cv2.imread(image_path)
    if image is not None:
        if is_overexposed(image):
            FileManager.move_file(image_path, overexposed_folder)
    else:
        logging.warning(f"Failed to read image: {image_path}")

def classify_overexposed_images(folder_path):
    """Classify all images in the given folder and move overexposed images to a new folder using multithreading."""
    image_files = [f for f in os.listdir(folder_path) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
    overexposed_folder = os.path.join(folder_path, 'overexposed')
    os.makedirs(overexposed_folder, exist_ok=True)

    # Determine the number of threads to use
    num_threads = min(32, os.cpu_count() + 4)  # Adjust based on your system

    with ThreadPoolExecutor(max_workers=num_threads) as executor:
        futures = []
        for filename in image_files:
            futures.append(executor.submit(process_image, filename, folder_path, overexposed_folder))

        # Optionally, wait for all futures to complete and handle exceptions
        for future in futures:
            try:
                future.result()
            except Exception as e:
                logging.error(f"Error processing image: {e}")

    FileManager.update_redo_file(folder_path, overexposed_folder)