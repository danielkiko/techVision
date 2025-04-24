import cv2
import numpy as np

def detect_vignette(image_path, threshold=0.3):
    # Конвертируем в grayscale и получаем размеры
    img = cv2.imread(image_path)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    h, w = gray.shape
    
    # Создаем маску центральной области (40% от ширины/высоты)
    center_mask = np.zeros((h, w), dtype=np.uint8)
    cv2.circle(center_mask, (w//2, h//2), int(min(w, h)*0.2), 255, -1)
    
    # Вычисляем среднюю яркость в центре и на краях
    center_mean = cv2.mean(gray, mask=center_mask)[0]
    edges_mean = cv2.mean(gray, mask=255-center_mask)[0]
    
    # Рассчитываем силу виньетирования
    vignette_strength = (center_mean - edges_mean) / center_mean if center_mean > 0 else 0
    
    # Определяем наличие виньетирования
    has_vignette = vignette_strength > threshold
    
    return vignette_strength, has_vignette
