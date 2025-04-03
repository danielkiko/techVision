import cv2
import numpy as np

def detect_glare(image_path, brightness_thresh=220, saturation_thresh=30):
    img = cv2.imread(image_path)
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    _, s, v = cv2.split(hsv)
    
    # Маска по яркости и насыщенности
    glare_mask = (v > brightness_thresh) & (s < saturation_thresh)
    glare_mask = glare_mask.astype(np.uint8) * 255
    
    # Улучшение маски
    kernel = np.ones((3, 3), np.uint8)
    glare_mask = cv2.morphologyEx(glare_mask, cv2.MORPH_CLOSE, kernel)
    cv2.imwrite("glare_mask.jpg", glare_mask)
    return glare_mask

def detect_glare_hsv(image_path, min_saturation=30, min_value=220):
    image = cv2.imread(image_path)
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    _, s, v = cv2.split(hsv)
    mask = (v > min_value) & (s < min_saturation)
    return mask.astype(np.uint8) * 255

