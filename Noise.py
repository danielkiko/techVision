import cv2
import numpy as np


def calculate_noise_variance(image_path):
    # Загрузка изображения в оттенках серого
    image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)

    # Вычисление градиента с помощью оператора Собеля
    sobel_x = cv2.Sobel(image, cv2.CV_64F, 1, 0, ksize=3)
    sobel_y = cv2.Sobel(image, cv2.CV_64F, 0, 1, ksize=3)

    # Вычисление магнитуды градиента
    gradient_magnitude = np.sqrt(sobel_x ** 2 + sobel_y ** 2)

    # Дисперсия магнитуды градиента
    noise_variance = gradient_magnitude.var()

    return noise_variance

def calculate_local_noise_variance(image_path, block_size=8):
    # Загрузка изображения в оттенках серого
    image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)

    # Разделение изображения на блоки
    h, w = image.shape
    local_variances = []

    for i in range(0, h, block_size):
        for j in range(0, w, block_size):
            block = image[i:i + block_size, j:j + block_size]
            if block.size > 0:
                local_variances.append(block.var())

    # Усреднение локальных дисперсий
    noise_level = np.mean(local_variances)

    return noise_level

def noise(image_path):
    noise_local_level = calculate_local_noise_variance(image_path)
    print(f"Noise Level (Local Variance): {noise_local_level}")
    noise_level = calculate_noise_variance(image_path)
    print(f"Noise Level (Variance): {noise_level}")
    #от нуля до ???
