import cv2
import numpy as np
import matplotlib.pyplot as plt
import argparse

def rolling_shutter_score(image_path,
                          min_line_length=50,
                          max_line_gap=5,
                          angle_margin=10,
                          show=False):
    img = cv2.imread(image_path)
    if img is None:
        raise FileNotFoundError(f"Cannot read image: {image_path}")
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(gray, 50, 150, apertureSize=3)

    # Найти отрезки
    lines = cv2.HoughLinesP(edges, 1, np.pi/180, threshold=80,
                            minLineLength=min_line_length,
                            maxLineGap=max_line_gap)
    if lines is None:
        raise RuntimeError("No lines detected for rolling shutter analysis.")

    tilts = []
    lengths = []
    for x1, y1, x2, y2 in lines.reshape(-1, 4):
        dx, dy = x2 - x1, y2 - y1
        angle = abs(np.degrees(np.arctan2(dy, dx)))  # 0..180°
        # отбросить горизонтальные и слишком косые
        if abs(90.0 - angle) <= angle_margin:
            tilt = abs(90.0 - angle)
            length = np.hypot(dx, dy)
            tilts.append(tilt)
            lengths.append(length)

    if not tilts:
        raise RuntimeError("No sufficiently vertical lines found.")

    # Взвешенное среднее отклонение от вертикали
    score = float(np.average(tilts, weights=lengths))

    if show:
        # Визуализация
        vis = cv2.cvtColor(edges, cv2.COLOR_GRAY2BGR)
        for (x1, y1, x2, y2), tilt in zip(lines.reshape(-1,4),
                                          [abs(90.0 - abs(np.degrees(np.arctan2(y2-y1, x2-x1)))) for x1,y1,x2,y2 in lines.reshape(-1,4)]):
            color = (0,255,0) if tilt <= angle_margin else (0,0,255)
            cv2.line(vis, (x1,y1), (x2,y2), color, 1)
        plt.figure(figsize=(8,5))
        plt.subplot(1,2,1)
        plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
        plt.title("Original")
        plt.axis('off')
        plt.subplot(1,2,2)
        plt.imshow(vis)
        plt.title(f"Rolling shutter tilt score: {score:.2f}°")
        plt.axis('off')
        plt.tight_layout()
        plt.show()

    return score

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Rolling Shutter artifact estimation"
    )
    parser.add_argument("image", help="Path to input image")
    parser.add_argument("--min_len", type=int, default=50,
                        help="Min line length for Hough (default=50)")
    parser.add_argument("--max_gap", type=int, default=5,
                        help="Max gap for Hough (default=5)")
    parser.add_argument("--angle_margin", type=float, default=10.0,
                        help="Tolerance around 90° for vertical lines (deg)")
    parser.add_argument("--show", action="store_true",
                        help="Display detected lines and score")
    args = parser.parse_args()

    score = rolling_shutter_score(
        args.image,
        min_line_length=args.min_len,
        max_line_gap=args.max_gap,
        angle_margin=args.angle_margin,
        show=args.show
    )
    print(f"Rolling shutter tilt score: {score:.2f}°")