import os
import shutil
import logging
import json
from snap_sort.utils.cache_manager import CacheManager
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

class FileManager:
    @classmethod
    def move_file(cls, src_path, dest_folder):
        if not os.path.exists(src_path):
            logging.error(f"Source file does not exist: {src_path}")
            return None
        
        os.makedirs(dest_folder, exist_ok=True)
        
        filename = os.path.basename(src_path)
        dest_path = os.path.join(dest_folder, filename)
        
        try:
            shutil.move(src_path, dest_path)
            logging.info(f"Moved {src_path} to {dest_path}")

            change = {"src": src_path, "dest": dest_path}
            cls.log_change(cls.get_redo_file_path(), change)

            return dest_path
        except Exception as e:
            logging.error(f"Failed to move {src_path} to {dest_path}: {e}")
            return None
    @classmethod
    def log_change(cls,log_file, change):
        with open(log_file, 'w') as f:
            json.dump(change, f)
        logging.info(f"Logged change: {change}")
    @classmethod
    def load_changes(cls,log_file):
        try:
            with open(log_file, 'r') as f:
                return json.load(f)
        except (json.JSONDecodeError, FileNotFoundError):
            return None