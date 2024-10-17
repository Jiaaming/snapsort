import os
import cv2
import logging
import json
import hashlib
from snap_sort.utils.cache_manager import CacheManager
from snap_sort.utils.constants import IMAGINE_CLASSES_FILE
from ultralytics import YOLO

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

class ImageLoader:
    cache_dir = CacheManager.get_cache_dir()

    @classmethod
    def get_image_hash(cls, image):
        image_bytes = cv2.imencode('.jpg', image)[1].tobytes() 
        return hashlib.md5(image_bytes).hexdigest()

    @classmethod
    def save_classes_to_file(cls, filename=IMAGINE_CLASSES_FILE):
        hash_file = os.path.join(cls.cache_dir, filename)
        with open(hash_file, "w") as f:
            json.dump(cls.image_classes, f)

    @classmethod
    def load_classes_from_file(cls, filename=IMAGINE_CLASSES_FILE):
        hash_file = os.path.join(cls.cache_dir, filename)
        try:
            with open(hash_file, "r") as f:
                cls.image_classes = json.load(f)
        except FileNotFoundError:
            cls.image_classes = {}

