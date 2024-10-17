import os
import hashlib
import tempfile
import logging
from snap_sort.utils.constants import SNAPSORT_CACHE, REDO_FILE
# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

class CacheManager:
    @classmethod
    def get_cache_dir(cls):
        xdg_runtime_dir = os.environ.get("XDG_RUNTIME_DIR")
        cache_dir = xdg_runtime_dir or os.path.join(tempfile.gettempdir(), SNAPSORT_CACHE)
        os.makedirs(cache_dir, exist_ok=True)
        return cache_dir
    
    @classmethod
    def get_file_hash(cls,file_path):
        hash_md5 = hashlib.md5()
        with open(file_path, "rb") as f:
            for chunk in iter(lambda: f.read(4096), b""):
                hash_md5.update(chunk)
        return hash_md5.hexdigest()
    @classmethod
    def get_cache_key(cls, image_path):
        file_hash = cls.get_file_hash(image_path)
        return file_hash

    @classmethod
    def clear_cache(cls):
        cache_dir = cls.get_cache_dir()
        for filename in os.listdir(cache_dir):
            file_path = os.path.join(cache_dir, filename)
            try:
                os.remove(file_path)
                logging.info(f"Removed cached file: {file_path}")
            except Exception as e:
                logging.error(f"Failed to remove cached file {file_path}: {e}")
    @classmethod
    def get_redo_file_path(cls):
        return os.path.join(cls.get_cache_dir(), REDO_FILE)