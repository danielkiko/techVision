import cv2
import numpy as np
import matplotlib.pyplot as plt
import argparse

def radial_distortion_coefficients(image_paths,
                                   pattern_size=(9, 6),
                                   square_size=1.0,
                                   show=False):

    objp = np.zeros((pattern_size[0]*pattern_size[1], 3), np.float32)
    objp[:, :2] = np.mgrid[0:pattern_size[0],
                           0:pattern_size[1]].T.reshape(-1, 2)
    objp *= square_size

    objpoints = []
    imgpoints = []
    img_shape = None

    for path in image_paths:
        img = cv2.imread(path)
        if img is None:
            print(f"[warn] файл не найден: {path}")
            continue
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        if img_shape is None:
            img_shape = gray.shape[::-1]

        found, corners = cv2.findChessboardCorners(gray,
                                                   pattern_size,
                                                   None)
        if not found:
            print(f"[warn] шахматка не найдена в {path}")
            continue

        # уточняем положения углов
        corners2 = cv2.cornerSubPix(gray, corners,
                                    (11, 11), (-1, -1),
                                    (cv2.TermCriteria_EPS +
                                     cv2.TermCriteria_MAX_ITER,
                                     30, 0.001))
        objpoints.append(objp)
        imgpoints.append(corners2)

        if show:
            vis = img.copy()
            cv2.drawChessboardCorners(vis, pattern_size, corners2, found)
            plt.figure(figsize=(5,4))
            plt.imshow(cv2.cvtColor(vis, cv2.COLOR_BGR2RGB))
            plt.title(f"Corners in {path}")
            plt.axis('off')
            plt.show()

    if not objpoints:
        raise RuntimeError("Не найдено ни одного узора шахматки.")

    # калибровка
    ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(
        objpoints, imgpoints, img_shape, None, None
    )
    if not ret:
        raise RuntimeError("Калибровка не удалась.")

    if show:
        print("Матрица камеры:\n", mtx)
        print("Коэффициенты дисторсии:\n", dist.ravel())

    return dist

def distortion_score_from_dist_coeffs(dist):

    k1, k2, p1, p2, k3 = dist.ravel()
    return abs(k1) + abs(k2) + abs(k3)

def estimate_edge_based_distortion(image_path, num_samples=2000, show=False):

    img = cv2.imread(image_path)
    if img is None:
        raise FileNotFoundError(f"{image_path} не найден")
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(gray, 50, 150, apertureSize=3)

    lines = cv2.HoughLinesP(edges, 1, np.pi/180, threshold=80,
                            minLineLength=50, maxLineGap=10)
    if lines is None:
        raise RuntimeError("Не удалось обнаружить достаточное число прямых.")

    deviations = []
    for x1, y1, x2, y2 in lines.reshape(-1, 4):
        # коэффициенты прямой ax + by + c = 0
        a = y2 - y1
        b = x1 - x2
        c = x2*y1 - x1*y2
        # длина
        length = np.hypot(a, b)
        if length == 0:
            continue
        # выбираем точки по отрезку
        xs = np.linspace(x1, x2, num=20)
        ys = np.linspace(y1, y2, num=20)
        # расстояние точки до прямой
        dist = np.abs(a*xs + b*ys + c) / length
        deviations.append(dist.mean())

    score = np.mean(deviations)
    if show:
        vis = cv2.cvtColor(edges, cv2.COLOR_GRAY2BGR)
        for x1, y1, x2, y2 in lines.reshape(-1, 4):
            cv2.line(vis, (x1, y1), (x2, y2), (0,255,0), 1)
        plt.figure(figsize=(5,5))
        plt.imshow(vis)
        plt.title(f"Edge distortion score: {score:.2f}")
        plt.axis('off')
        plt.show()

    return score

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Geometric Distortion Analysis"
    )
    parser.add_argument("images", nargs="+",
                        help="Калибровочные или тестовые изображения")
    parser.add_argument("--pattern_size", type=int, nargs=2,
                        default=[9, 6],
                        help="Размер шахматки (корнеры по X, по Y)")
    parser.add_argument("--square_size", type=float, default=1.0,
                        help="Размер квадрата для калибровки")
    parser.add_argument("--show", action="store_true",
                        help="Показывать промежуточные результаты")
    args = parser.parse_args()

    # Расчёт по шахматке (если поданы >1 изображения)
    dist = radial_distortion_coefficients(
        args.images,
        pattern_size=tuple(args.pattern_size),
        square_size=args.square_size,
        show=args.show
    )
    score_radial = distortion_score_from_dist_coeffs(dist)
    print(f"Radial distortion score: {score_radial:.6f}")

    # Оценка по краям на первом изображении
    edge_score = estimate_edge_based_distortion(
        args.images[0], show=args.show
    )
    print(f"Edge-based distortion score: {edge_score:.4f}")