import os
import shutil
import logging
from snap_sort.utils.file_manager import FileManager
from snap_sort.utils.cache_manager import CacheManager
import json

logging.getLogger().setLevel(logging.WARNING)

def redo_last_operation():
    """
    Undo the last operation by moving files back to their original locations.
    """
    change = load_changes(CacheManager.get_redo_file_path())
    if not change:
        return

    src_dir = change['dest']
    dest_dir = change['src']
    # Check if the source directory exists
    if not os.path.exists(src_dir):
        logging.info(f"Source directory does not exist: {src_dir}")
        return
    if not os.path.exists(dest_dir):
        logging.info(f"Destination directory does not exist: {dest_dir}")
        return
    
    try:
        shutil.copytree(
            src_dir,
            dest_dir,
            dirs_exist_ok=True,
            copy_function=shutil.move
        )
        shutil.rmtree(src_dir)
    except Exception as e:
        logging.error(f"Failed to move files from {src_dir} back to {dest_dir}: {e}")

    # Clear the changes log after completing the redo operation
    logging.info("Redo completed successfully.")

def load_changes(log_file):
    try:
        with open(log_file, 'r') as f:
            content = f.read().strip()
            if not content:
                return []  # Return an empty list if the file is empty
            return json.loads(content)
    except json.JSONDecodeError as e:
        logging.error(f"Error decoding JSON from {log_file}: {e}")
        return []
def clear_changes(log_file):
    with open(log_file, 'w') as f:
        f.write('')