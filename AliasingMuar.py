import cv2
import numpy as np
import matplotlib.pyplot as plt

def aliasing_score(image_path, show_spectrum=False):

    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    f = np.fft.fft2(img)
    fshift = np.fft.fftshift(f)
    magnitude_spectrum = 20 * np.log(np.abs(fshift) + 1)

    h, w = img.shape
    center = (h // 2, w // 2)
    radius = min(h, w) // 4

    Y, X = np.ogrid[:h, :w]
    dist_from_center = np.sqrt((X - center[1]) ** 2 + (Y - center[0]) ** 2)
    high_freq_mask = dist_from_center > radius

    high_freq_energy = magnitude_spectrum[high_freq_mask].mean()
    low_freq_energy = magnitude_spectrum[~high_freq_mask].mean()
    score = high_freq_energy / (low_freq_energy + 1e-6)

    if show_spectrum:
        plt.figure(figsize=(10, 4))
        plt.subplot(1, 2, 1)
        plt.imshow(img, cmap='gray')
        plt.title('Image')
        plt.axis('off')
        plt.subplot(1, 2, 2)
        plt.imshow(magnitude_spectrum, cmap='gray')
        plt.title('Magnitude Spectrum')
        plt.axis('off')
        plt.show()

    print(f'Aliasing score: {score:.2f}')
    return score

def detect_moire(image_path, show_result=False):

    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    # Параметры фильтра Габора
    ksize = 31
    sigma = 4.0
    lambd = 10.0
    gamma = 0.5
    psi = 0
    moire_score = 0

    # Применяем фильтры Габора с разными углами
    for theta in np.arange(0, np.pi, np.pi / 4):
        kernel = cv2.getGaborKernel((ksize, ksize), sigma, theta, lambd, gamma, psi, ktype=cv2.CV_32F)
        filtered = cv2.filter2D(img, cv2.CV_8UC3, kernel)
        moire_score = max(moire_score, filtered.std())

        if show_result:
            plt.imshow(filtered, cmap='gray')
            plt.title(f'Gabor theta={theta:.2f}')
            plt.axis('off')
            plt.show()

    print(f'Moire score: {moire_score:.2f}')
    return moire_score

# Пример использования:
if __name__ == "__main__":
    image_path = "Images_for_run/moire_example.png"
    aliasing_score(image_path, show_spectrum=True)
    detect_moire(image_path, show_result=True)