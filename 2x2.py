import cv2
import numpy as np
import os
import glob
import re
from itertools import permutations
from collections import deque

BASE_OUTPUT = r"C:\Users\aliha\VSC code\Image"

OUT_IMG = os.path.join(BASE_OUTPUT, "final_output")
OUT_ASSEMBLED = os.path.join(BASE_OUTPUT, "assembled")

os.makedirs(OUT_ASSEMBLED, exist_ok=True)
os.makedirs(os.path.join(OUT_ASSEMBLED, "2x2"), exist_ok=True)

_piece_re = re.compile(r"piece[_\-]?(\d+)\.\w+$", re.IGNORECASE)


def read_pieces_sorted(piece_dir):
    files = glob.glob(os.path.join(piece_dir, "*.png"))
    indexed = []
    for f in files:
        m = _piece_re.search(os.path.basename(f))
        if not m:
            continue
        idx = int(m.group(1))
        img = cv2.imread(f, cv2.IMREAD_COLOR)
        if img is None:
            continue
        indexed.append((idx, img))
    if not indexed:
        return []
    indexed.sort(key=lambda x: x[0])
    return [img for _, img in indexed]


def extract_borders(piece, strip_width=8):
    lab = cv2.cvtColor(piece, cv2.COLOR_BGR2LAB).astype(np.float32)
    h, w = lab.shape[:2]
    sw = min(strip_width, h//2, w//2)

    def make_grad_patch_enhanced(patch_lab):
        patch_bgr = cv2.cvtColor(patch_lab.astype(np.uint8), cv2.COLOR_LAB2BGR).astype(np.float32)
        patch_gray = cv2.cvtColor(patch_bgr.astype(np.uint8), cv2.COLOR_BGR2GRAY).astype(np.float32)
        patch_gray = cv2.GaussianBlur(patch_gray, (3,3), 0)
        
        gx = cv2.Sobel(patch_gray, cv2.CV_32F, 1, 0, ksize=3) 
        gy = cv2.Sobel(patch_gray, cv2.CV_32F, 0, 1, ksize=3)

        grad_mag = cv2.magnitude(gx, gy)[..., None]
        grad_dir = cv2.phase(gx, gy, angleInDegrees=True)[..., None]
        lap = cv2.Laplacian(patch_gray, cv2.CV_32F)[..., None]

        return np.concatenate([patch_lab, grad_mag, grad_dir, lap], axis=2)

    return {
        0: make_grad_patch_enhanced(lab[0:sw, :, :]),
        1: make_grad_patch_enhanced(lab[:, w-sw:w, :]),
        2: make_grad_patch_enhanced(lab[h-sw:h, :, :]),
        3: make_grad_patch_enhanced(lab[:, 0:sw, :])
    }


def normalize_strip_2d(strip):
    arr = strip.astype(np.float32)
    for ch in range(arr.shape[2]):
        m, sd = arr[..., ch].mean(), arr[..., ch].std()
        arr[..., ch] = (arr[..., ch] - m) / (sd if sd > 1e-6 else 1.0)
    return arr


def border_distance_2d(stripA, stripB, sideA, sideB, p=0.3, q=1/16,
                        w_color=0.4, w_grad_mag=0.2, w_grad_dir=0.2, w_lap=0.2):
    def orient(s, side):
        return normalize_strip_2d(np.transpose(s, (1,0,2)) if side in (1,3) else s)

    a, b = orient(stripA, sideA), orient(stripB, sideB)
    if a.size == 0 or b.size == 0:
        return 1e9
    if a.shape[:2] != b.shape[:2]:
        b = cv2.resize(b, (a.shape[1], a.shape[0]), interpolation=cv2.INTER_LINEAR)

    def dist(x, y):
        d_color = np.sum(np.abs(x[...,:3] - y[...,:3])**p)
        d_grad_mag = np.sum(np.abs(x[...,3:4] - y[...,3:4])**p)
        d_grad_dir = np.sum(np.abs(x[...,4:5] - y[...,4:5])**p)
        d_lap = np.sum(np.abs(x[...,5:6] - y[...,5:6])**p)
        total = (w_color*d_color + w_grad_mag*d_grad_mag + w_grad_dir*d_grad_dir + w_lap*d_lap)
        return total**(q/p)

    return float(dist(a, b))


def build_compatibility(pieces, strip_width=8):
    n = len(pieces)
    borders = [extract_borders(p, strip_width) for p in pieces]
    compat = {s: np.full((n,n), 1e9, dtype=np.float32) for s in range(4)}
    
    for i in range(n):
        for j in range(n):
            if i == j:
                continue
            compat[0][i,j] = border_distance_2d(borders[i][0], borders[j][2], 0, 2)
            compat[1][i,j] = border_distance_2d(borders[i][1], borders[j][3], 1, 3)
            compat[2][i,j] = border_distance_2d(borders[i][2], borders[j][0], 2, 0)
            compat[3][i,j] = border_distance_2d(borders[i][3], borders[j][1], 3, 1)
    return compat


def solve_bruteforce(pieces, compat, grid_n):
    n = grid_n * grid_n
    best_perm = None
    best_score = 1e12
    
    for perm in permutations(range(n)):
        score = 0.0
        valid = True
        for pos, pid in enumerate(perm):
            r = pos // grid_n
            c = pos % grid_n
            if c > 0:
                left_pid = perm[pos-1]
                score += compat[1][left_pid, pid]
                if score >= best_score:
                    valid = False
                    break
            if r > 0:
                top_pid = perm[pos-grid_n]
                score += compat[2][top_pid, pid]
                if score >= best_score:
                    valid = False
                    break
        if not valid:
            continue
        if score < best_score:
            best_score = score
            best_perm = perm
    return list(best_perm), best_score


def assemble_from_pieces(piece_dir, grid=2, strip_width=8):
    indexed = read_pieces_sorted(piece_dir)
    if not indexed:
        raise ValueError(f"No piece images found in: {piece_dir}")

    needed = grid * grid
    if len(indexed) < needed:
        raise ValueError(f"Found {len(indexed)} pieces in {piece_dir}, need {needed}")

    pieces = indexed[:needed]
    tile_h, tile_w = pieces[0].shape[:2]
    
    print("Building compatibility matrices...")
    compat = build_compatibility(pieces, strip_width=strip_width)
    
    print("Solving 2x2 by brute-force permutations...")
    order, score = solve_bruteforce(pieces, compat, grid)
    
    assembled = np.zeros((tile_h * grid, tile_w * grid, 3), dtype=pieces[0].dtype)
    for idx, piece_idx in enumerate(order):
        r, c = divmod(idx, grid)
        assembled[r*tile_h:(r+1)*tile_h, c*tile_w:(c+1)*tile_w] = pieces[piece_idx]
    
    return assembled


def main():
    results = []
    total = 0

    for folder in sorted(os.listdir(OUT_IMG)):
        folder_path = os.path.join(OUT_IMG, folder)
        if not os.path.isdir(folder_path):
            continue
        if "2x2" not in folder.lower():
            continue

        for image_basename in sorted(os.listdir(folder_path)):
            image_dir = os.path.join(folder_path, image_basename)
            if not os.path.isdir(image_dir):
                continue

            try:
                assembled = assemble_from_pieces(image_dir, grid=2, strip_width=8)
            except Exception as e:
                print(f"[WARN] Skipping '{image_basename}' in '{folder}': {e}")
                continue

            assembled_out_path = os.path.join(OUT_ASSEMBLED, "2x2", f"{image_basename}_assembled.png")
            cv2.imwrite(assembled_out_path, assembled)
            total += 1

            results.append({
                "basename": image_basename,
                "folder": folder,
                "assembled_path": assembled_out_path,
            })

            print(f"[INFO] Assembled '{image_basename}' -> {assembled_out_path}")

    if total == 0:
        print("[RESULT] No images assembled (no 2x2 pieces found).")
        return

    print(f"\n===== SUMMARY =====")
    print(f"Total images assembled : {total}")
    print(f"Output directory       : {os.path.join(OUT_ASSEMBLED, '2x2')}")
    print("=====================")


if __name__ == "__main__":
    main()
