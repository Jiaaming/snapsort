import os
import cv2
import imagehash
from PIL import Image
from snap_sort.utils.file_manager import FileManager
from snap_sort.utils.image_loader import ImageLoader
from heapq import heappush, heappop


def calculate_phash_similarity(image1, image2):
    """Calculate similarity using perceptual hash."""
    hash1 = imagehash.phash(Image.fromarray(cv2.cvtColor(image1, cv2.COLOR_BGR2RGB)))
    hash2 = imagehash.phash(Image.fromarray(cv2.cvtColor(image2, cv2.COLOR_BGR2RGB)))
    # Calculate the Hamming distance
    distance = hash1 - hash2
    # Convert distance to similarity (0.0 to 1.0)
    max_distance = 64  # Maximum distance for 64-bit hash
    similarity = 1 - (distance / max_distance)
    return similarity

def calculate_histogram_similarity(image1, image2):
    """Calculate similarity using histogram comparison."""
    # Convert images to HSV color space
    hsv1 = cv2.cvtColor(image1, cv2.COLOR_BGR2HSV)
    hsv2 = cv2.cvtColor(image2, cv2.COLOR_BGR2HSV)
    # Calculate the histogram for each image
    hist1 = cv2.calcHist([hsv1], [0, 1], None, [50, 60], [0, 180, 0, 256])
    hist2 = cv2.calcHist([hsv2], [0, 1], None, [50, 60], [0, 180, 0, 256])
    # Normalize the histograms
    cv2.normalize(hist1, hist1)
    cv2.normalize(hist2, hist2)
    # Compare histograms using correlation
    similarity = cv2.compareHist(hist1, hist2, cv2.HISTCMP_CORREL)
    return similarity

def find_similar_images(reference_image_path, folder_path, top_n=10, weight_phash=0.5, weight_hist=0.5):
    """Find the top N most similar images using a combination of perceptual hash and histogram comparison."""
    similarities_heap = []

    reference_image = ImageLoader.resize_image(cv2.imread(reference_image_path))

    for filename in os.listdir(folder_path):
        if filename.lower().endswith(('.png', '.jpg', '.jpeg')):
            image_path = os.path.join(folder_path, filename)
            
            image = ImageLoader.resize_image(cv2.imread(image_path))
            
            phash_similarity = calculate_phash_similarity(reference_image, image)
            hist_similarity = calculate_histogram_similarity(reference_image, image)
            
            combined_similarity = (weight_phash * phash_similarity) + (weight_hist * hist_similarity)
            if combined_similarity >= 0.4:
                heappush(similarities_heap, (-combined_similarity, filename))
                if len(similarities_heap) > top_n:
                    heappop(similarities_heap)

    similar_folder = os.path.join(folder_path, 'similar')
    os.makedirs(similar_folder, exist_ok=True)

    for similarity_score,filename in similarities_heap:
        src_path = os.path.join(folder_path, filename)
        FileManager.move_file(src_path, similar_folder)
    FileManager.update_redo_file(folder_path, similar_folder)
    return len(similarities_heap)
