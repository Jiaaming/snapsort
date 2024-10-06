import cv2
import os
import click

@click.command()
@click.argument('folder_path')
def classify(folder_path):
    """分类文件夹中的照片"""
    classify_images_in_folder(folder_path)

if __name__ == "__main__":
    classify()


def load_image(image_path):
    image = cv2.imread(image_path)
    if image is None:
        raise ValueError(f"Could not load image at {image_path}")
    return image

def is_overexposed(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    # 计算图像的亮度直方图
    hist = cv2.calcHist([gray], [0], None, [256], [0, 256])
    # 判断是否过曝
    return hist[-1] > 0.1 * image.size  # 假设超过 10% 的像素是过亮的

def classify_images_in_folder(folder_path):
    for filename in os.listdir(folder_path):
        if filename.endswith(('.png', '.jpg', '.jpeg')):
            image_path = os.path.join(folder_path, filename)
            image = load_image(image_path)
            if is_overexposed(image):
                # 移动到过曝文件夹
                os.makedirs('overexposed', exist_ok=True)
                os.rename(image_path, f"overexposed/{filename}")


