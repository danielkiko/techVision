import cv2
import numpy as np
import matplotlib.pyplot as plt
import argparse

def light_power_score(image_path,
                      grid_size=(3, 3),
                      show=False):
    img = cv2.imread(image_path)
    if img is None:
        raise FileNotFoundError(f"Не удалось загрузить {image_path}")
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY).astype(np.float32)
    h, w = gray.shape
    cols, rows = grid_size

    # Разбиение на ячейки
    step_x = w // cols
    step_y = h // rows
    grid_means = np.zeros((rows, cols), dtype=np.float32)

    for i in range(rows):
        y0, y1 = i * step_y, h if i == rows-1 else (i+1) * step_y
        for j in range(cols):
            x0, x1 = j * step_x, w if j == cols-1 else (j+1) * step_x
            cell = gray[y0:y1, x0:x1]
            grid_means[i, j] = cell.mean()

    mean_lum = float(grid_means.mean())
    min_lum = float(grid_means.min())
    max_lum = float(grid_means.max())
    uniformity = min_lum / max_lum if max_lum > 0 else 0.0
    coeff_var = float(grid_means.std() / mean_lum) if mean_lum > 0 else 0.0

    if show:
        # Визуализация оригинала с сеткой
        vis = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        fig, axes = plt.subplots(1, 2, figsize=(10, 5))

        ax1 = axes[0]
        ax1.imshow(vis)
        ax1.set_title("Original image with grid")
        for i in range(1, rows):
            y = int(i * step_y)
            ax1.axhline(y, color='w', linewidth=1)
        for j in range(1, cols):
            x = int(j * step_x)
            ax1.axvline(x, color='w', linewidth=1)
        ax1.axis('off')

        # Тепловая карта средних яркостей
        ax2 = axes[1]
        im = ax2.imshow(grid_means, cmap='inferno',
                        vmin=grid_means.min(), vmax=grid_means.max())
        ax2.set_title("Mean luminance per cell")
        ax2.set_xticks(np.arange(cols))
        ax2.set_yticks(np.arange(rows))
        for (i, j), val in np.ndenumerate(grid_means):
            ax2.text(j, i, f"{val:.1f}", ha='center', va='center',
                     color='white', fontsize=8)
        fig.colorbar(im, ax=ax2, shrink=0.8)
        ax2.axis('off')

        plt.suptitle(
            f"Mean={mean_lum:.1f}, Uniformity={uniformity:.3f}, "
            f"Cv={coeff_var:.3f}", fontsize=12
        )
        plt.tight_layout(rect=[0, 0, 1, 0.93])
        plt.show()

    return mean_lum, uniformity, coeff_var, grid_means

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Light Power: средняя яркость и uniformity"
    )
    parser.add_argument("image", help="Путь к входному изображению")
    parser.add_argument("-g", "--grid", type=int, nargs=2,
                        metavar=("COLS", "ROWS"),
                        default=[3, 3],
                        help="Сетка для оценки (cols rows), default=3 3")
    parser.add_argument("--show", action="store_true",
                        help="Визуализация результатов")
    args = parser.parse_args()

    mean_lum, uniformity, coeff_var, _ = light_power_score(
        args.image,
        grid_size=tuple(args.grid),
        show=args.show
    )
    print(f"Mean luminance  : {mean_lum:.2f}")
    print(f"Uniformity (min/max) : {uniformity:.3f}")
    print(f"Coeff. of variation  : {coeff_var:.3f}")