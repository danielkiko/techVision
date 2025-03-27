import cv2
import numpy as np
import matplotlib.pyplot as plt

def colors(image_path):
    # Загрузка изображения
    image = cv2.imread(image_path)

    # Разделение на каналы R, G, B
    b, g, r = cv2.split(image)

    # Вычисление средних значений каналов
    mean_b = np.mean(b)
    mean_g = np.mean(g)
    mean_r = np.mean(r)

    print(f"Mean B: {mean_b}, Mean G: {mean_g}, Mean R: {mean_r}")

    # Оценка баланса белого
    if abs(mean_r - mean_g) > 10 or abs(mean_r - mean_b) > 10 or abs(mean_g - mean_b) > 10:
        print("Баланс белого нарушен.")
    else:
        print("Баланс белого нормальный.")

def colors2(image_path):
    image = cv2.imread(image_path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    # Построение гистограммы для каждого канала
    colors = ('r', 'g', 'b')
    for i, color in enumerate(colors):
        histogram = cv2.calcHist([image], [i], None, [256], [0, 256])
        plt.plot(histogram, color=color)
        plt.xlim([0, 256])

    plt.title('Color Histogram')
    plt.show()

    #На графике снизу показаны яркость пикселя
    #Слева - количество пикселей данной яркость (цвета)

    #Если ничего не понял закинь белую и черную картинки

# def colors_ohvat_old(image_path):
#     # Загрузка изображения
#     image = cv2.imread(image_path)
#
#     # Преобразование в цветовое пространство CIE XYZ
#     image_xyz = colour.RGB_to_XYZ(image, colour.sRGB_COLOURSPACE.whitepoint, colour.sRGB_COLOURSPACE.whitepoint,
#                                   colour.sRGB_COLOURSPACE.matrix_RGB_to_XYZ)
#
#     # Оценка цветового охвата
#     gamut = colour.RGB_Colourspace(colour.sRGB_COLOURSPACE.name, colour.sRGB_COLOURSPACE.primaries,
#                                    colour.sRGB_COLOURSPACE.whitepoint)
#     print(f"Цветовой охват: {gamut}")

def colors_ohvat(image_path):
    # Загрузка изображения
    image = cv2.imread(image_path)

    # Преобразование изображения в формат RGB
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    # Изменение формы массива пикселей в двумерный (количество_пикселей x 3 канала)
    pixels = image_rgb.reshape(-1, 3)

    # Вычисление уникальных цветов
    unique_colors = np.unique(pixels, axis=0)

    # Количество уникальных цветов
    num_unique_colors = unique_colors.shape[0]

    print(f'Количество уникальных цветов в изображении: {num_unique_colors}')