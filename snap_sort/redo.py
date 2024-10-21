import os
import shutil
import logging
from snap_sort.utils.file_manager import FileManager
from snap_sort.utils.cache_manager import CacheManager
import json


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
        logging.error(f"Source directory does not exist: {src_dir}")
        return
    if not os.path.exists(dest_dir):
        logging.error(f"Destination directory does not exist: {dest_dir}")
        return
    
    try:
        # Create the destination directory if it does not exist
        os.makedirs(dest_dir, exist_ok=True)

        # Move all files from src_dir to dest_dir
        for filename in os.listdir(src_dir):
            src_file = os.path.join(src_dir, filename)
            dest_file = os.path.join(dest_dir, filename)
            
            # Move the file
            shutil.move(src_file, dest_file)
        
        # Remove the source directory after moving all files
        os.rmdir(src_dir)
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