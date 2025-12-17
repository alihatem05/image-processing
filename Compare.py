import cv2
import os
import numpy as np
import json
import time

<<<<<<< HEAD
LOG_PATH = r"D:\Gam3a\Junior\Fall 25\Image\Gravity Falls\.cursor"


=======
LOG_PATH = r"e:\Studying\Fall 25\CSE381\Project\Gravity Falls\.cursor"

    
>>>>>>> 7aabc83 (Updating Readme file and Adding the dataset and output)
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

<<<<<<< HEAD
BASE_OUTPUT = r"D:\Gam3a\Junior\Fall 25\Image\Gravity Falls"
=======
BASE_OUTPUT = r"e:\Studying\Fall 25\CSE381\Project\Gravity Falls"
>>>>>>> 7aabc83 (Updating Readme file and Adding the dataset and output)

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
<<<<<<< HEAD

=======
    
>>>>>>> 7aabc83 (Updating Readme file and Adding the dataset and output)
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
<<<<<<< HEAD
























=======
def extract_id(name):
    return os.path.splitext(name)[0].replace("_assembled", "")
def main():
    print("=" * 60)
    print("PUZZLE ACCURACY EVALUATION")
    print("=" * 60)

    for grid, assembled_dir in PUZZLE_FOLDERS.items():
        n_tiles = int(grid[0])
        print(f"\n[{grid}]")

        if not os.path.exists(assembled_dir):
            print(f"  ❌ Assembled folder not found: {assembled_dir}")
            continue

        assembled_imgs = [
            f for f in os.listdir(assembled_dir)
            if f.endswith(".png")
        ]

        if not assembled_imgs:
            print("  ❌ No assembled images found")
            continue

        total_acc = 0
        valid = 0

        for img_name in assembled_imgs:
            puzzle_id = extract_id(img_name)
            correct_path = os.path.join(CORRECT_FOLDER, f"{puzzle_id}.png")
            assembled_path = os.path.join(assembled_dir, img_name)

            if not os.path.exists(correct_path):
                print(f"  ⚠ Missing correct image for {puzzle_id}")
                continue

            reconstructed = cv2.imread(assembled_path)
            if reconstructed is None:
                print(f"  ⚠ Failed to read {img_name}")
                continue

            acc = calc_accuracy(reconstructed, correct_path, n_tiles)
            print(f"  {puzzle_id}: {acc:.2f}%")

            total_acc += acc
            valid += 1

        if valid > 0:
            print(f"  ▶ Average {grid} accuracy: {total_acc / valid:.2f}%")
        else:
            print("  ❌ No valid comparisons")

    print("\nDONE.")
if __name__ == "__main__":
    main()
>>>>>>> 7aabc83 (Updating Readme file and Adding the dataset and output)
