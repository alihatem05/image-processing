import cv2
import os
import numpy as np

BASE_OUTPUT = r"D:\Gam3a\Junior\Fall 25\Image\Gravity Falls"

DATASET = os.path.join(BASE_OUTPUT, "dataset", "Gravity Falls")
OUT_ASSEMBLED = os.path.join(BASE_OUTPUT, "assembled", "4x4")
CORRECT_FOLDER = os.path.join(DATASET, "correct")

PUZZLE_FOLDERS = {
    "2x2": os.path.join(BASE_OUTPUT, "assembled", "2x2"),
    "4x4": OUT_ASSEMBLED,
    "8x8": os.path.join(BASE_OUTPUT, "assembled", "8x8")
}

def calc_accuracy(reconstructed, correct_path, n_tiles):
    correct_img = cv2.imread(correct_path)
    if correct_img is None:
        return 0.0

    h, w = correct_img.shape[:2]
    tile_h, tile_w = h // n_tiles, w // n_tiles

    correct_tiles = []
    for r in range(n_tiles):
        for c in range(n_tiles):
            y1, y2 = r * tile_h, min((r + 1) * tile_h, h)
            x1, x2 = c * tile_w, min((c + 1) * tile_w, w)
            correct_tiles.append(correct_img[y1:y2, x1:x2])

    rh, rw = reconstructed.shape[:2]
    rec_tiles = []
    for r in range(n_tiles):
        for c in range(n_tiles):
            y1, y2 = r * (rh // n_tiles), (r + 1) * (rh // n_tiles)
            x1, x2 = c * (rw // n_tiles), (c + 1) * (rw // n_tiles)
            rec_tiles.append(reconstructed[y1:y2, x1:x2])

    match_count = 0
    tile_scores = []
    
    if n_tiles == 2:
        threshold = 0.6
    elif n_tiles == 4:
        threshold = 0.7
    else:
        threshold = 0.75
    
    for tile_idx, (rec_tile, corr_tile) in enumerate(zip(rec_tiles, correct_tiles)):
        if rec_tile.size == 0 or corr_tile.size == 0:
            continue
        
        if rec_tile.shape != corr_tile.shape:
            corr_tile = cv2.resize(corr_tile, (rec_tile.shape[1], rec_tile.shape[0]))
        
        diff = cv2.absdiff(rec_tile, corr_tile)
        score = 1 - np.sum(diff) / diff.size / 255
        tile_scores.append(score)
        
        is_match = score > threshold
        if is_match:
            match_count += 1
    
    accuracy = (match_count / (n_tiles ** 2)) * 100
























