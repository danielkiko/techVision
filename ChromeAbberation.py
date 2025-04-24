import cv2
import numpy as np
import matplotlib.pyplot as plt
import argparse

def lateral_chromatic_aberration(image_path, show=False):
    img = cv2.imread(image_path)
    if img is None:
        raise FileNotFoundError(f"Image not found: {image_path}")
    b, g, r = cv2.split(img)

    edges_r = cv2.Canny(r, 50, 150)
    edges_g = cv2.Canny(g, 50, 150)
    edges_b = cv2.Canny(b, 50, 150)

    shift_rg, _ = cv2.phaseCorrelate(np.float32(edges_r), np.float32(edges_g))
    shift_bg, _ = cv2.phaseCorrelate(np.float32(edges_b), np.float32(edges_g))

    score_rg = np.hypot(*shift_rg)
    score_bg = np.hypot(*shift_bg)
    score = score_rg + score_bg

    if show:
        overlay = cv2.merge([edges_b, edges_g, edges_r])
        plt.figure(figsize=(5,5))
        plt.imshow(overlay)
        plt.title("Edge overlay (B=blue, G=green, R=red)")
        plt.axis('off')
        plt.show()

        print(f"Shift R→G: dx={shift_rg[0]:.2f}, dy={shift_rg[1]:.2f}")
        print(f"Shift B→G: dx={shift_bg[0]:.2f}, dy={shift_bg[1]:.2f}")

    return score, shift_rg, shift_bg

def longitudinal_chromatic_aberration(image_path, show=False):
    img = cv2.imread(image_path)
    if img is None:
        raise FileNotFoundError(f"Image not found: {image_path}")
    b, g, r = cv2.split(img)

    lap_r = cv2.Laplacian(r, cv2.CV_64F)
    lap_g = cv2.Laplacian(g, cv2.CV_64F)
    lap_b = cv2.Laplacian(b, cv2.CV_64F)

    var_r = lap_r.var()
    var_g = lap_g.var()
    var_b = lap_b.var()

    diff_rg = abs(var_r - var_g)
    diff_bg = abs(var_b - var_g)
    score = diff_rg + diff_bg

    if show:
        plt.figure(figsize=(6,4))
        bins = 50
        plt.hist(lap_r.ravel(), bins=bins, alpha=0.5, label=f"R var={var_r:.2f}", color='r')
        plt.hist(lap_g.ravel(), bins=bins, alpha=0.5, label=f"G var={var_g:.2f}", color='g')
        plt.hist(lap_b.ravel(), bins=bins, alpha=0.5, label=f"B var={var_b:.2f}", color='b')
        plt.legend()
        plt.title("Laplacian Histograms per Channel")
        plt.show()

        print(f"Variance R={var_r:.2f}, G={var_g:.2f}, B={var_b:.2f}")

    return score, (var_r, var_g, var_b)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Chromatic Aberration Analysis")
    parser.add_argument("image", help="Path to input image")
    parser.add_argument("--show", action="store_true",
                        help="Show intermediate plots (edges, histograms)")
    args = parser.parse_args()

    print("=== Lateral Chromatic Aberration ===")
    lat_score, shift_rg, shift_bg = lateral_chromatic_aberration(args.image, args.show)
    print(f"Lateral CA score: {lat_score:.2f}\n")

    print("=== Longitudinal Chromatic Aberration ===")
    long_score, variances = longitudinal_chromatic_aberration(args.image, args.show)
    print(f"Longitudinal CA score: {long_score:.2f}")