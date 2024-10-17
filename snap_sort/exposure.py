# overexposure.py
import cv2
import os
import logging
from snap_sort.utils.file_manager import FileManager
from snap_sort.utils.image_loader import ImageLoader
from concurrent.futures import ThreadPoolExecutor
from snap_sort.utils.constants import IMAGINE_CLASSES_FILE

def is_overexposed(image):
    """Check if the image is overexposed by analyzing a central region of interest (ROI)."""
    image = ImageLoader.resize_image(image)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    hist = cv2.calcHist([gray], [0], None, [256], [0, 256])
    
    return hist[-1] > 0.1 * image.size


def classify_overexposed_images(folder_path):
    """Classify all images in the given folder and move overexposed images to a new folder."""
    image_files = [f for f in os.listdir(folder_path) if f.endswith(('.png', '.jpg', '.jpeg'))]

    for filename in image_files:
        image_path = os.path.join(folder_path, filename)
        logging.info(f"Classifying image: {image_path}")
        image = cv2.imread(image_path)
        if is_overexposed(image):
            overexposed_folder = os.path.join(folder_path, 'overexposed')
            FileManager.move_file(image_path, overexposed_folder)