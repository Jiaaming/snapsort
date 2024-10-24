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

tokenizer = AutoTokenizer.from_pretrained('prajjwal1/bert-mini')
model = AutoModel.from_pretrained('prajjwal1/bert-mini')


def semantic_search_images(prompt, folder_path, top_n=10):
    """Find the top N most semantically similar images using a pre-trained model."""
    image_files = [f for f in os.listdir(folder_path) if f.endswith(('.png', '.jpg', '.jpeg'))]
    hash_results_map = load_hash_results_from_json()
    import faiss

    dimension = 384

    embeddings_list = []
    image_paths = []

    new_folder_path = os.path.join(folder_path, prompt)
    for filename in image_files:
        image_path = os.path.join(folder_path, filename)
        image = cv2.imread(image_path)
        image_hash = ImageLoader.get_image_hash(image)

        results = []
        if image_hash not in hash_results_map:
            results = perform_yolo_detection(image_path)
            hash_results_map[image_hash] = (results, image_path)
        else:
            results = hash_results_map[image_hash][0]
            old_path = hash_results_map[image_hash][1]
            if old_path != image_path:
                hash_results_map[image_hash] = (results, image_path)

        combined_results = " ".join(results)
        embedding = get_embeddings([combined_results])[0]

        embeddings_list.append(embedding)
        image_paths.append(image_path)
    faiss_index = faiss.IndexFlatL2(dimension)
    embeddings_matrix = np.array(embeddings_list)
    faiss_index.add(embeddings_matrix)

    prompt_embedding = get_embeddings([prompt])[0].reshape(1, -1)
    distances, indices = faiss_index.search(prompt_embedding, top_n)

    similar_image_paths = [image_paths[idx] for idx in indices[0]]
    for image_path in similar_image_paths:
        logging.info(f"Found similar image: {image_path}")
        FileManager.move_file(image_path, new_folder_path)
    save_hash_results_to_json(hash_results_map)
    FileManager.update_redo_file(folder_path, new_folder_path)

def get_embeddings(texts):
    try:
        from sentence_transformers import SentenceTransformer
    except ImportError:
        logging.error("Please install the sentence-transformers library: pip install sentence-transformers")
        return None
    sbert_model = SentenceTransformer('all-MiniLM-L6-v2')

    return sbert_model.encode(texts)

# hash_results_map = {
#     "hash1": (["dog", "cat"], "/path/to/image1.jpg"),
#     "hash2": (["car", "person"], "/path/to/image2.jpg")
# }
def save_hash_results_to_json(hash_results_map):
    json_file_path = os.path.join(CacheManager.get_cache_dir(), HASH_CLASSES_MAP)
    serializable_map = {key: {"yolo_results": value[0], "file_path": value[1]} for key, value in
                        hash_results_map.items()}
    print("serializable_map: ", serializable_map)
    with open(json_file_path, 'w') as f:
        json.dump(serializable_map, f)


def load_hash_results_from_json():
    json_file_path = os.path.join(CacheManager.get_cache_dir(), HASH_CLASSES_MAP)
    if not os.path.exists(json_file_path):
        return {}
    with open(json_file_path, 'r') as f:
        serializable_map = json.load(f)

    hash_results_map = {key: (value["yolo_results"], value["file_path"]) for key, value in serializable_map.items()}
    return hash_results_map


def perform_yolo_detection(image_path):
    try:
        model_path = os.path.join(os.path.dirname(__file__), 'models', 'yolov8s.pt')
        model = YOLO(model_path, verbose=False)
    except Exception as e:
        logging.error(f"Failed to load YOLO model: {e}")
        return []
    results = model(image_path)
    classes = []
    for result in results:
        boxes = result.boxes
        names = result.names

        for box in boxes:
            confidence = box.conf[0].item()
            if confidence > 0.30:
                cls_id = int(box.cls[0])
                class_name = names[cls_id]
                classes.append(class_name)
                print(f"Detected {class_name} with confidence {confidence:.2f}")
    return classes


