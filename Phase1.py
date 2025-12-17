import cv2
import numpy as np
import os
import glob

BASE_OUTPUT = "."

DATASET = os.path.join(BASE_OUTPUT, "dataset", "Gravity Falls")

OUT_IMG = os.path.join(BASE_OUTPUT, "final_output")
OUT_PRE = os.path.join(BASE_OUTPUT, "preprocess")

os.makedirs(OUT_IMG, exist_ok=True)
os.makedirs(OUT_PRE, exist_ok=True)

def get_grid(folder_name):
    if "2x2" in folder_name: return 2
    if "4x4" in folder_name: return 4
    if "8x8" in folder_name: return 8
    return None

def preprocess_steps(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    blur = cv2.GaussianBlur(gray, (3, 3), 0)

    laplacian = cv2.Laplacian(blur, cv2.CV_16S, ksize=3)
    laplacian_abs = cv2.convertScaleAbs(laplacian)

    final = cv2.addWeighted(blur, 0.85, laplacian_abs, 0.15, 0)

    return {
        "gray": gray,
        "blur": blur,
        "final": final
    }


img_global_id = 1

for folder in os.listdir(DATASET):
    folder_path = os.path.join(DATASET, folder)
    if folder.lower() == "correct" or not os.path.isdir(folder_path):
        continue

    grid = get_grid(folder.lower())
    if grid is None:
        continue

    print(f"[INFO] Folder: {folder} | Grid: {grid}x{grid}")
    images = glob.glob(os.path.join(folder_path, "*.*"))

    for path in images:
        img = cv2.imread(path)
        if img is None:
            continue

        img_basename = os.path.splitext(os.path.basename(path))[0]

        pre_out_dir = os.path.join(OUT_PRE, f"{grid}x{grid}", img_basename)
        os.makedirs(pre_out_dir, exist_ok=True)

        steps = preprocess_steps(img)

        cv2.imwrite(os.path.join(pre_out_dir, "1_gray.png"), steps["gray"])
        cv2.imwrite(os.path.join(pre_out_dir, "2_blur.png"), steps["blur"])
        cv2.imwrite(os.path.join(pre_out_dir, "3_final.png"), steps["final"])

        h, w = img.shape[:2]
        tile_h = h // grid
        tile_w = w // grid

        grid_folder = os.path.join(OUT_IMG, folder)
        os.makedirs(grid_folder, exist_ok=True)

        image_out_dir = os.path.join(grid_folder, img_basename)
        os.makedirs(image_out_dir, exist_ok=True)

        pieceID = 1
        pieces_from_original = 0
        for row in range(grid):
            for col in range(grid):
                y1, y2 = row * tile_h, (row + 1) * tile_h
                x1, x2 = col * tile_w, (col + 1) * tile_w
                piece_img = img[y1:y2, x1:x2]
                pieces_from_original += 1
                out_path = os.path.join(image_out_dir, f"piece_{pieceID}.png")
                cv2.imwrite(out_path, piece_img)
                pieceID += 1

        print(f"[INFO] {pieceID-1} tiles saved for '{img_basename}' (grid: {grid}x{grid})")
        img_global_id += 1
