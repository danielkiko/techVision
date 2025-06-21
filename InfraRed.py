import cv2
import numpy as np
import matplotlib.pyplot as plt
import argparse

def ir_contamination_score(image_path, threshold=10, show=False):
    img = cv2.imread(image_path)
    if img is None:
        raise FileNotFoundError(f"Не удалось прочитать изображение: {image_path}")

    b, g, r = cv2.split(img.astype(np.float32))
    ir_diff = r - (g + b) / 2.0
    ir_diff = np.clip(ir_diff, 0, None)

    mean_score = float(ir_diff.mean())
    mask = ir_diff > threshold
    pct_above = float(mask.sum()) / mask.size * 100.0

    if show:
        plt.figure(figsize=(12,4))
        plt.subplot(1,3,1)
        plt.imshow(cv2.cvtColor(img.astype(np.uint8), cv2.COLOR_BGR2RGB))
        plt.title("Original")
        plt.axis("off")

        plt.subplot(1,3,2)
        plt.imshow(ir_diff, cmap="inferno")
        plt.title(f"IR Diff (mean={mean_score:.1f})")
        plt.colorbar(shrink=0.7)
        plt.axis("off")

        plt.subplot(1,3,3)
        plt.imshow(mask, cmap="gray")
        plt.title(f"Mask > {threshold} ({pct_above:.1f}%)")
        plt.axis("off")

        plt.tight_layout()
        plt.show()

    print(f"IR diff = {mean_score}")
    print(f"Mask % = {pct_above}")
    return mean_score, pct_above, ir_diff, mask

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="InfraRed contamination estimation"
    )
    parser.add_argument("image", help="Путь к входному изображению")
    parser.add_argument("-t", "--threshold", type=float, default=10.0,
                        help="Порог IR-дифференциала (default=10)")
    parser.add_argument("--show", action="store_true",
                        help="Показывать графики (IR-карта, маска)")
    args = parser.parse_args()

    mean_score, pct_above, _, _ = ir_contamination_score(
        args.image, threshold=args.threshold, show=args.show
    )
    print(f"InfraRed mean diff: {mean_score:.2f}")
    print(f"Pixels above {args.threshold}: {pct_above:.1f}%")