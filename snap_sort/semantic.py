import os
import logging
from ultralytics import YOLO
from snap_sort.utils.file_manager import FileManager
from snap_sort.utils.image_loader import ImageLoader
from snap_sort.utils.constants import HASH_CLASSES_MAP
from snap_sort.utils.cache_manager import CacheManager
import cv2
from transformers import AutoTokenizer, AutoModel
import numpy as np
import json
logging.getLogger().setLevel(logging.WARNING)
import sys
import contextlib

with open(os.devnull, 'w') as f, contextlib.redirect_stdout(f):
    tokenizer = AutoTokenizer.from_pretrained('prajjwal1/bert-mini')
    bert_model = AutoModel.from_pretrained('prajjwal1/bert-mini')
    # Determine the absolute path to the model directory
    current_dir = os.path.dirname(os.path.abspath(__file__))
    model_path = os.path.join(current_dir, 'models', 'all-MiniLM-L6-v2')

    try:
        from sentence_transformers import SentenceTransformer
    except ImportError:
        logging.error("Please install the sentence-transformers library: pip install sentence-transformers")
        sbert_model = None
    else:
        # Load the model from the local path
        sbert_model = SentenceTransformer(model_path)


def semantic_search_images(prompt, folder_path, top_n=10):
    """Find the top N most semantically similar images using a pre-trained model."""
    image_files = [f for f in os.listdir(folder_path) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
    hash_results_map = load_hash_results_from_json()
    dimension = 384  # Adjust based on your embedding dimension
    embeddings_list = []
    image_paths = []
    new_folder_path = os.path.join(folder_path, prompt)

    # Load images that need detection
    images_info = load_images_to_detect(image_files, folder_path, hash_results_map)
    images_to_detect = images_info['images_to_detect']
    images_to_detect_hashes = images_info['images_to_detect_hashes']
    embeddings_list.extend(images_info['embeddings_list'])
    image_paths.extend(images_info['image_paths'])

    # Perform YOLO detection in batches
    if images_to_detect:
        perform_batch_yolo_detection(
            images_to_detect,
            images_to_detect_hashes,
            hash_results_map,
            embeddings_list,
            image_paths
        )

    # Build the Faiss index and search for similar images
    similar_image_paths = search_similar_images(
        embeddings_list,
        image_paths,
        prompt,
        dimension,
        top_n
    )

    # Move similar images to the new folder
    for image_path in similar_image_paths:
        FileManager.move_file(image_path, new_folder_path)

    save_hash_results_to_json(hash_results_map)
    FileManager.update_redo_file(folder_path, new_folder_path)

def load_images_to_detect(image_files, folder_path, hash_results_map):
    """Load images and identify which ones need YOLO detection."""
    images_to_detect = []
    images_to_detect_hashes = []
    embeddings_list = []
    image_paths = []

    for filename in image_files:
        image_path = os.path.join(folder_path, filename)
        image = cv2.imread(image_path)
        image_hash = ImageLoader.get_image_hash(image)

        if image_hash not in hash_results_map:
            images_to_detect.append(image_path)
            images_to_detect_hashes.append(image_hash)
        else:
            # Update path if necessary
            results, old_path = hash_results_map[image_hash]
            if old_path != image_path:
                hash_results_map[image_hash] = (results, image_path)
            if results:
                embedding = extract_embedding_from_results(results)
                embeddings_list.append(embedding)
                image_paths.append(image_path)

    return {
        'images_to_detect': images_to_detect,
        'images_to_detect_hashes': images_to_detect_hashes,
        'embeddings_list': embeddings_list,
        'image_paths': image_paths
    }

def perform_batch_yolo_detection(images_to_detect, images_to_detect_hashes, hash_results_map, embeddings_list, image_paths):
    """Perform YOLO detection on images in batches."""
    # Load the YOLO model once
    try:
        model_path = os.path.join(os.path.dirname(__file__), 'models', 'yolov8s.pt')
        model = YOLO(model_path, verbose=False)
    except Exception as e:
        logging.error(f"Failed to load YOLO model: {e}")
        return

    # Determine batch size automatically (you can adjust this logic as needed)
    total_images = len(images_to_detect)
    max_batch_size = 16  # Set a maximum batch size based on your system capacity
    batch_size = min(total_images, max_batch_size)

    # Process images in batches
    for i in range(0, total_images, batch_size):
        batch_image_paths = images_to_detect[i:i + batch_size]
        batch_image_hashes = images_to_detect_hashes[i:i + batch_size]

        # YOLOv8 accepts a list of image paths
        results_list = model(batch_image_paths)

        for image_path, image_hash, results in zip(batch_image_paths, batch_image_hashes, results_list):
            classes = parse_yolo_results(results)
            hash_results_map[image_hash] = (classes, image_path)

            if classes:
                embedding = extract_embedding_from_results(classes)
                embeddings_list.append(embedding)
                image_paths.append(image_path)

def parse_yolo_results(results):
    """Parse YOLO detection results to extract class names."""
    classes = []
    boxes = results.boxes
    names = results.names

    for box in boxes:
        confidence = box.conf[0].item()
        if confidence > 0.30:
            cls_id = int(box.cls[0])
            class_name = names[cls_id]
            classes.append(class_name)
    if not classes:
        return None
    return classes

def extract_embedding_from_results(results):
    """Extract embeddings from YOLO detection results."""
    combined_results = " ".join(results)
    embedding = get_embeddings([combined_results])[0]
    return embedding

def search_similar_images(embeddings_list, image_paths, prompt, dimension, top_n):
    """Search for similar images using Faiss index."""
    if not embeddings_list:
        logging.warning("No embeddings found to build Faiss index.")
        return []
    import faiss
    # Build the Faiss index
    faiss_index = faiss.IndexFlatL2(dimension)
    embeddings_matrix = np.array(embeddings_list).astype('float32')
    faiss_index.add(embeddings_matrix)

    # Search for similar images
    prompt_embedding = get_embeddings([prompt])[0].reshape(1, -1).astype('float32')
    distances, indices = faiss_index.search(prompt_embedding, min(top_n, len(embeddings_list)))
    similar_image_paths = [image_paths[idx] for idx in indices[0]]
    return similar_image_paths

def update_hash_results_map(hash_results_map, image_hash, classes, image_path):
    """Update the hash results map with new detection results."""
    hash_results_map[image_hash] = (classes, image_path)

def get_embeddings(texts):
    if sbert_model is None:
        logging.error("SentenceTransformer model is not loaded.")
        return None
    return sbert_model.encode(texts)

# hash_results_map = {
#     "hash1": (["dog", "cat"], "/path/to/image1.jpg"),
#     "hash2": (["car", "person"], "/path/to/image2.jpg")
# }
def save_hash_results_to_json(hash_results_map):
    json_file_path = os.path.join(CacheManager.get_cache_dir(), HASH_CLASSES_MAP)
    serializable_map = {key: {"yolo_results": value[0], "file_path": value[1]} for key, value in
                        hash_results_map.items()}
    # print("serializable_map: ", serializable_map)
    with open(json_file_path, 'w') as f:
        json.dump(serializable_map, f)


def load_hash_results_from_json():
    json_file_path = os.path.join(CacheManager.get_cache_dir(), HASH_CLASSES_MAP)
    if not os.path.exists(json_file_path):
        return {}
    with open(json_file_path, 'r') as f:
        serializable_map = json.load(f)

    hash_results_map = {key: (value["yolo_results"], value["file_path"]) for key, value in serializable_map.items()}
    logging.info(f"Loaded {len(hash_results_map)} hash results from JSON.")
    return hash_results_map




