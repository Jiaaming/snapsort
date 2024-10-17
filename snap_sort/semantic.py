import os
import logging
from ultralytics import YOLO

from snap_sort.utils.file_manager import FileManager
from snap_sort.utils.image_loader import ImageLoader
from snap_sort.utils.constants import HASH_CLASSES_MAP
from snap_sort.utils.cache_manager import CacheManager
import cv2
from transformers import AutoTokenizer, AutoModel
import torch
import numpy as np
import faiss
import json

model_path = os.path.join(os.path.dirname(__file__), 'models', 'yolov8s.pt')
model = YOLO(model_path, verbose=False)

tokenizer = AutoTokenizer.from_pretrained('prajjwal1/bert-mini')
model = AutoModel.from_pretrained('prajjwal1/bert-mini')


# 假设我们有以下文本
texts = ["dog","light", "person", "cat", "car", "apple", "banana"]

# 获取文本嵌入
embeddings = get_embeddings(texts)

# 设置嵌入的维度 (TinyBERT 的向量维度通常为 768)
dimension = embeddings.shape[1]

# 创建 FAISS 索引
index = faiss.IndexFlatL2(dimension)  # 使用 L2 距离的索引
# 将嵌入向量添加到索引中
index.add(embeddings)

# 查询文本 "animal"
query_text = "animal"
query_embedding = get_embeddings([query_text])[0]

# 在 FAISS 中查找最相似的 3 个文本
k = 3  # 查找 3 个最相似的
distances, indices = index.search(np.array([query_embedding]), k)

# 打印结果
print(f"Query: {query_text}")
for idx, distance in zip(indices[0], distances[0]):
    print(f"Nearest text: {texts[idx]} with distance: {distance}")


def semantic_search_images(prompt, folder_path, top_n=10):
    """Find the top N most semantically similar images using a pre-trained model."""
    image_files = [f for f in os.listdir(folder_path) if f.endswith(('.png', '.jpg', '.jpeg'))]
    hash_results_map = load_hash_results_from_json()

    # FAISS 索引设置
    d = 384  # 嵌入维度 (根据模型决定, 这里用 all-MiniLM-L6-v2 的默认维度)
    faiss_index = faiss.IndexFlatL2(d)  # 使用 L2 距离的索引

    embeddings_list = []  # 存储所有嵌入
    image_paths = []  # 存储所有图片路径

    new_folder_path = os.path.join(folder_path, prompt)
    for filename in image_files:
        image_path = os.path.join(folder_path, filename)
        image = cv2.imread(image_path)
        image_hash = ImageLoader.get_image_hash(image)

        results = []
        if image_hash not in hash_results_map:
            results = perform_yolo_detection(image)
            hash_results_map[image_hash] = (results, image_path)
        else:
            results = hash_results_map[image_hash][0]
            old_path = hash_results_map[image_hash][1]
            if old_path != image_path:
                logging.info(f"Updating cache path from {old_path} to {image_path}")
                hash_results_map[image_hash] = (results, image_path)
                # 将 YOLO 检测结果（类别）转换为嵌入

        combined_results = " ".join(results)  # 例如将 YOLO 类别列表合并为一个字符串
        embedding = get_embeddings([combined_results])[0]  # 生成嵌入

        embeddings_list.append(embedding)
        image_paths.append(image_path)

    embeddings_matrix = np.array(embeddings_list)
    faiss_index.add(embeddings_matrix)

    prompt_embedding = get_embeddings([prompt])[0].reshape(1, -1)
    distances, indices = faiss_index.search(prompt_embedding, top_n)

    similar_image_paths = [image_paths[idx] for idx in indices[0]]
    for image_path in similar_image_paths:
        logging.info(f"Found similar image: {image_path}")
        FileManager.move_file(image_path, new_folder_path)
    save_hash_results_to_json(hash_results_map)

def get_embeddings(texts):
    inputs = tokenizer(texts, return_tensors='pt', padding=True, truncation=True)
    with torch.no_grad():
        outputs = model(**inputs)
    embeddings = outputs.last_hidden_state.mean(dim=1)
    return embeddings.numpy()


# hash_results_map = {
#     "hash1": (["dog", "cat"], "/path/to/image1.jpg"),
#     "hash2": (["car", "person"], "/path/to/image2.jpg")
# }
def save_hash_results_to_json(hash_results_map):
    json_file_path = os.path.join(CacheManager.get_cache_dir(), HASH_CLASSES_MAP)
    serializable_map = {key: {"yolo_results": value[0], "file_path": value[1]} for key, value in
                        hash_results_map.items()}

    with open(json_file_path, 'w') as f:
        json.dump(serializable_map, f)


def load_hash_results_from_json():
    json_file_path = os.path.join(CacheManager.get_cache_dir(), HASH_CLASSES_MAP)

    with open(json_file_path, 'r') as f:
        serializable_map = json.load(f)

    hash_results_map = {key: (value["yolo_results"], value["file_path"]) for key, value in serializable_map.items()}
    return hash_results_map


def perform_yolo_detection(cls, image):
    results = cls.model(image)
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

    return classes
