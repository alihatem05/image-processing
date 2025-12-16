import cv2
import os
import numpy as np
import json
import time

LOG_PATH = r"D:\Gam3a\Junior\Fall 25\Image\Gravity Falls\.cursor"


def debug_log(location, message, data=None, hypothesis_id=None, run_id="initial"):
    try:
        log_entry = {
            "sessionId": "debug-session",
            "runId": run_id,
            "timestamp": int(time.time() * 1000),
            "location": location,
            "message": message,
            "data": data or {}
        }
        if hypothesis_id:
            log_entry["hypothesisId"] = hypothesis_id
        with open(LOG_PATH, "a", encoding="utf-8") as f:
            f.write(json.dumps(log_entry) + "\n")
    except Exception:
        pass

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
    
    debug_log("Compare.py:calc_accuracy", "Starting tile comparison", {"n_tiles": n_tiles, "total_tiles": n_tiles ** 2, "threshold": threshold}, "L1", "initial")
    
    for tile_idx, (rec_tile, corr_tile) in enumerate(zip(rec_tiles, correct_tiles)):
        if rec_tile.size == 0 or corr_tile.size == 0:
            debug_log("Compare.py:calc_accuracy", "Empty tile detected", {"tile_idx": tile_idx}, "L1", "initial")
            continue
        
        debug_log("Compare.py:calc_accuracy", "Comparing tile", {"tile_idx": tile_idx, "rec_shape": rec_tile.shape, "corr_shape": corr_tile.shape}, "L2", "initial")
        
        if rec_tile.shape != corr_tile.shape:
            corr_tile = cv2.resize(corr_tile, (rec_tile.shape[1], rec_tile.shape[0]))
            debug_log("Compare.py:calc_accuracy", "Resized correct tile", {"new_shape": corr_tile.shape}, "L2", "initial")
        
        diff = cv2.absdiff(rec_tile, corr_tile)
        score = 1 - np.sum(diff) / diff.size / 255
        tile_scores.append(score)
        
        is_match = score > threshold
        if is_match:
            match_count += 1
        
        debug_log("Compare.py:calc_accuracy", "Tile comparison result", {"tile_idx": tile_idx, "score": float(score), "threshold": threshold, "is_match": is_match, "tile_size": rec_tile.size}, "L2", "initial")
    
    accuracy = (match_count / (n_tiles ** 2)) * 100
    debug_log("Compare.py:calc_accuracy", "Accuracy calculated", {"match_count": match_count, "total_tiles": n_tiles ** 2, "accuracy": float(accuracy), "min_score": float(min(tile_scores)) if tile_scores else 0, "max_score": float(max(tile_scores)) if tile_scores else 0, "avg_score": float(np.mean(tile_scores)) if tile_scores else 0}, "L1", "initial")
    
    return accuracy
























