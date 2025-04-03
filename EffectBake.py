import cv2
import numpy as np

def blur_degree(image_path):
    img = cv2.imread(image_path)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    laplacian = cv2.Laplacian(gray, cv2.CV_64F)
    print(f"Blur score: {laplacian.var()}")  # Чем выше, тем резче
    return laplacian.var()


def bokeh_estimation(image_path):
    img = cv2.imread(image_path)
    mask = np.zeros(img.shape[:2], np.uint8)
    bgd_model = np.zeros((1, 65), np.float64)
    fgd_model = np.zeros((1, 65), np.float64)
    rect = (50, 50, img.shape[1]-100, img.shape[0]-100)  # Примерный ROI
    cv2.grabCut(img, mask, rect, bgd_model, fgd_model, 5, cv2.GC_INIT_WITH_RECT)
    
    # Маска фона (0 или 2)
    background_mask = np.where((mask == 0) | (mask == 2), 1, 0).astype('uint8')
    background = cv2.bitwise_and(img, img, mask=background_mask)
    
    # Оценка размытости фона
    background_blur = blur_degree(background)
    foreground_blur = blur_degree(cv2.bitwise_and(img, img, mask=1-background_mask))
    print(f"Bokeh effect score: {background_blur / foreground_blur}")
    return background_blur / foreground_blur  # Чем больше, тем сильнее боке



def detect_bokeh_shapes(image_path, min_radius=5, max_radius=50):
    img = cv2.imread(image_path)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (11, 11), 0)
    circles = cv2.HoughCircles(
        blurred, cv2.HOUGH_GRADIENT, dp=1, minDist=20,
        param1=50, param2=30, minRadius=min_radius, maxRadius=max_radius
    )
    
    if circles is not None:
        circles = np.uint16(np.around(circles))
        for i in circles[0, :]:
            cv2.circle(img, (i[0], i[1]), i[2], (0, 255, 0), 2)
        print(f"Detected bokeh circles: {len(circles[0])}")
        return len(circles[0])  # Кол-во кругов (чем больше, тем лучше боке)
    return 0

