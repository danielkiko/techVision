import cv2
import numpy as np


def calculate_sharpness(image_path):
    # Загрузка изображения в оттенках серого
    image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)

    # Вычисление градиента с помощью оператора Лапласа
    laplacian = cv2.Laplacian(image, cv2.CV_64F)

    # Вычисление "резкости" как среднего значения абсолютного градиента
    sharpness = np.mean(np.abs(laplacian))

    return sharpness

def sharp():
    # Пример использования
    image_path = "D:\\qwerty.png"
    sharpness = calculate_sharpness(image_path)
    print(f"Sharpness: {sharpness}")