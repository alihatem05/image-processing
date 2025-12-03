import cv2
import numpy as np
import os
import glob

DATASET = os.path.join("dataset", "Gravity Falls")
OUT_IMG = "final_output"
OUT_PRE = "preprocess"

os.makedirs(OUT_IMG, exist_ok=True)
os.makedirs(OUT_PRE, exist_ok=True)

def get_grid(folder_name):
    if "2x2" in folder_name: return 2
    if "4x4" in folder_name: return 4
    if "8x8" in folder_name: return 8
    return None

def preprocess_steps(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    blur = cv2.GaussianBlur(gray, (5,5), 0)

    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8,8))
    enh = clahe.apply(blur)

    sharp_kernel = np.array([[0,-1,0], [-1,5,-1], [0,-1,0]])
    sharp = cv2.filter2D(enh, -1, sharp_kernel)

    ga = cv2.GaussianBlur(sharp, (55,55), 0)
    final = cv2.addWeighted(sharp, 1.5, ga, -0.5, 0)

    return {"gray": gray, "blur": blur, "clahe": enh, "sharp": sharp, "final": final}

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
        cv2.imwrite(os.path.join(pre_out_dir, "3_clahe.png"), steps["clahe"])
        cv2.imwrite(os.path.join(pre_out_dir, "4_sharp.png"), steps["sharp"])
        cv2.imwrite(os.path.join(pre_out_dir, "5_final.png"), steps["final"])

        enhanced_image = steps["final"]

        _, mask = cv2.threshold(enhanced_image, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7,7))
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=2)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=2)

        h, w = img.shape[:2]
        tile_h = h // grid
        tile_w = w // grid

        grid_folder = os.path.join(OUT_IMG, f"{grid}x{grid}")
        os.makedirs(grid_folder, exist_ok=True)

        image_out_dir = os.path.join(grid_folder, img_basename)
        os.makedirs(image_out_dir, exist_ok=True)

        pieceID = 1

        for row in range(grid):
            for col in range(grid):
                y1, y2 = row * tile_h, (row + 1) * tile_h
                x1, x2 = col * tile_w, (col + 1) * tile_w

                piece_img = img[y1:y2, x1:x2]

                out_name = f"piece_{pieceID}.png"
                out_path = os.path.join(image_out_dir, out_name)
                cv2.imwrite(out_path, piece_img)

                pieceID += 1

        print(f"{pieceID-1} tiles saved successfully for image '{img_basename}' (folder: {grid}x{grid}).")
        img_global_id += 1
