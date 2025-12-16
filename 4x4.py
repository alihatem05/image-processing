from collections import deque

import cv2
import numpy as np
import os
import glob
import re

BASE_OUTPUT = r"D:\Gam3a\Junior\Fall 25\Image\Gravity Falls"
OUT_IMG = os.path.join(BASE_OUTPUT, "final_output")
OUT_ASSEMBLED = os.path.join(BASE_OUTPUT, "assembled", "4x4")

os.makedirs(OUT_ASSEMBLED, exist_ok=True)

_piece_re = re.compile(r"piece[_\-]?(\d+)\.\w+$", re.IGNORECASE)


def preprocess_piece(img):
    if len(img.shape) == 3:
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    else:
        gray = img.copy()
    
    blurred = cv2.GaussianBlur(gray, (3, 3), 0)
    
    laplacian = cv2.Laplacian(blurred, cv2.CV_16S, ksize=3)
    laplacian_abs = cv2.convertScaleAbs(laplacian)
    
    final = cv2.addWeighted(blurred, 0.85, laplacian_abs, 0.15, 0)
    
    return final


def read_pieces_sorted(piece_dir):






    files = glob.glob(os.path.join(piece_dir, "*.png"))
    indexed = []
    for f in files:
        m = _piece_re.search(os.path.basename(f))
        if not m:
            continue
        idx = int(m.group(1))
        img = cv2.imread(f)
        if img is not None:
            indexed.append((idx, img))
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

    print("  Computing compatibility scores...")
    for i in range(n):
        if (i + 1) % 4 == 0:
            print(f"    Progress: {i+1}/{n} pieces processed")
        for j in range(n):
            if i == j:
                continue
            compat[0][i,j] = border_distance_2d(borders[i][0], borders[j][2], 0, 2)
            compat[1][i,j] = border_distance_2d(borders[i][1], borders[j][3], 1, 3)
            compat[2][i,j] = border_distance_2d(borders[i][2], borders[j][0], 2, 0)
            compat[3][i,j] = border_distance_2d(borders[i][3], borders[j][1], 3, 1)
    return compat



def opposite(side):
    return (side + 2) % 4

def best_partner_for(i, side, compat):
    arr = compat[side][i]
    return int(np.argmin(arr))

def is_best_buddy(i, side, j, compat):
    if i == j:
        return False
    bj = best_partner_for(i, side, compat)
    if bj != j:
        return False
    opp = opposite(side)
    bi = best_partner_for(j, opp, compat)
    return bi == i



def placer(n, grid_n, compat, seed_placement=None, seed_center=True):
    placement = [-1] * n
    used = [False] * n

    def mutual_best_buddy(a, side_a, b):
        best_for_a = np.argmin(compat[side_a][a])
        best_for_b = np.argmin(compat[opposite(side_a)][b])
        return best_for_a == b and best_for_b == a

    if seed_placement:
        seed_pos = list(seed_placement.keys())
        rs = [p // grid_n for p in seed_pos]
        cs = [p % grid_n for p in seed_pos]
        rmin, rmax = min(rs), max(rs)
        cmin, cmax = min(cs), max(cs)
        seed_h = rmax - rmin + 1
        seed_w = cmax - cmin + 1

        if seed_center:
            top = (grid_n - seed_h) // 2
            left = (grid_n - seed_w) // 2
        else:
            top, left = 0, 0

        for pos_old, pid in seed_placement.items():
            r_old, c_old = pos_old // grid_n, pos_old % grid_n
            r_new = top + (r_old - rmin)
            c_new = left + (c_old - cmin)
            if 0 <= r_new < grid_n and 0 <= c_new < grid_n:
                pos_new = r_new * grid_n + c_new
                placement[pos_new] = pid
                used[pid] = True
    else:
        seed_pid = np.random.randint(0, n)
        seed_pos = np.random.choice(range(n))
        placement[seed_pos] = seed_pid
        used[seed_pid] = True

    def get_neighbors(pos):
        r, c = pos // grid_n, pos % grid_n
        neighbors = []
        if r > 0 and placement[pos - grid_n] != -1:
            neighbors.append((pos - grid_n, 2))
        if r < grid_n - 1 and placement[pos + grid_n] != -1:
            neighbors.append((pos + grid_n, 0))
        if c > 0 and placement[pos - 1] != -1:
            neighbors.append((pos - 1, 1))
        if c < grid_n - 1 and placement[pos + 1] != -1:
            neighbors.append((pos + 1, 3))
        return neighbors

    slots_filled = sum(1 for x in placement if x != -1)
    while slots_filled < n:
        empty_slots = []
        for pos in range(n):
            if placement[pos] != -1:
                continue
            neighs = get_neighbors(pos)
            if neighs:
                empty_slots.append((-len(neighs), pos, neighs))
        if not empty_slots:
            pos = placement.index(-1)
            empty_slots = [(0, pos, [])]

        empty_slots.sort()
        chosen = None

        for _, slot_pos, neighs in empty_slots:
            candidates = []
            for pid in range(n):
                if used[pid]:
                    continue
                bb_count = 0
                compat_sum = 0.0
                for neigh_pos, neigh_side in neighs:
                    neigh_pid = placement[neigh_pos]
                    if mutual_best_buddy(neigh_pid, neigh_side, pid):
                        bb_count += 1
                    compat_sum += compat[neigh_side][neigh_pid, pid]
                if bb_count > 0:
                    candidates.append((bb_count, compat_sum, slot_pos, pid))
            if candidates:
                candidates.sort(key=lambda x: (-x[0], x[1]))
                chosen = candidates[0]
                break

        if chosen is None:
            _, slot_pos, neighs = empty_slots[0]
            best_val = 1e18
            best_pid = None
            for pid in range(n):
                if used[pid]:
                    continue
                ssum = 0.0
                for neigh_pos, neigh_side in neighs:
                    neigh_pid = placement[neigh_pos]
                    ssum += compat[neigh_side][neigh_pid, pid]
                avg = ssum / max(1, len(neighs))
                if avg < best_val:
                    best_val = avg
                    best_pid = pid
            chosen = (0, best_val, slot_pos, best_pid)

        _, _, slot_pos, chosen_pid = chosen
        placement[slot_pos] = chosen_pid
        used[chosen_pid] = True
        slots_filled += 1

    return placement



def segmenter(placement, grid_n, compat):
    n_slots = len(placement)
    visited = [False] * n_slots
    segments = []

    def neighbors(pos):
        r = pos // grid_n
        c = pos % grid_n
        if c > 0: yield pos-1, 3
        if c < grid_n-1: yield pos+1, 1
        if r > 0: yield pos-grid_n, 0
        if r < grid_n-1: yield pos+grid_n, 2

    for pos in range(n_slots):
        if visited[pos]:
            continue
        queue = deque([pos])
        comp = []
        visited[pos] = True
        while queue:
            u = queue.popleft()
            comp.append(u)
            pu = placement[u]
            for v, side_of_u in neighbors(u):
                if visited[v]:
                    continue
                pv = placement[v]
                if is_best_buddy(pu, side_of_u, pv, compat):
                    visited[v] = True
                    queue.append(v)
        if comp:
            segments.append(comp)
    return segments



def compute_best_buddies_score(placement, grid_n, compat):
    n_slots = len(placement)
    bb_count = 0
    total_adj = 0
    for pos in range(n_slots):
        r = pos // grid_n
        c = pos % grid_n
        pid = placement[pos]
        if c < grid_n - 1:
            np0 = pos + 1
            pid2 = placement[np0]
            total_adj += 1
            if is_best_buddy(pid, 1, pid2, compat):
                bb_count += 1
        if r < grid_n - 1:
            np1 = pos + grid_n
            pid2 = placement[np1]
            total_adj += 1
            if is_best_buddy(pid, 2, pid2, compat):
                bb_count += 1
    if total_adj == 0:
        return 0.0
    return bb_count / total_adj



def shifter(initial_placement, grid_n, compat, max_iters=10, swap_pass=True):
    n_slots = len(initial_placement)
    current = initial_placement.copy()
    best_score = compute_best_buddies_score(current, grid_n, compat)

    for it in range(max_iters):
        segments = segmenter(current, grid_n, compat)
        if not segments:
            break

        segments.sort(key=lambda x: -len(x))
        improved = False

        for seg in segments:
            if len(seg) == 0:
                continue
            seed_map = {pos: current[pos] for pos in seg}
            placement_new = placer(n_slots, grid_n, compat, seed_placement=seed_map)
            score_new = compute_best_buddies_score(placement_new, grid_n, compat)

            if score_new > best_score + 1e-9:
                current = placement_new
                best_score = score_new
                improved = True
                break

        if not improved and swap_pass:
            for pos1 in range(n_slots):
                for pos2 in range(pos1+1, n_slots):
                    new_placement = current.copy()
                    new_placement[pos1], new_placement[pos2] = new_placement[pos2], new_placement[pos1]
                    score_swap = compute_best_buddies_score(new_placement, grid_n, compat)
                    if score_swap > best_score + 1e-9:
                        current = new_placement
                        best_score = score_swap
                        improved = True
                        break
                if improved:
                    break

        if not improved:
            break

    return current, best_score



def assemble_4x4(piece_dir, strip_width=8, seeds=5, shifter_iters=8):
    pieces = read_pieces_sorted(piece_dir)
    assert len(pieces) >= 16, f"Need at least 16 pieces, got {len(pieces)}"

    pieces = pieces[:16]
    grid = 4
    needed = 16
    tile_h, tile_w = pieces[0].shape[:2]

    print("Phase 1: Building compatibility matrices...")
    compat = build_compatibility(pieces, strip_width=strip_width)

    best_placement = None
    best_bb_score = -1.0
    
    for s in range(seeds):
        print(f"  Seed run {s+1}/{seeds}")
        init_placement = placer(needed, grid, compat, seed_placement=None)
        bb0 = compute_best_buddies_score(init_placement, grid, compat)

        placement_after_shifter, bb_sh = shifter(init_placement, grid, compat, max_iters=shifter_iters)

        if bb_sh >= bb0:
            final_placement = placement_after_shifter
            final_bb = bb_sh
        else:
            final_placement = init_placement
            final_bb = bb0

        print(f"    BB-score after seed {s+1}: {final_bb:.4f}")
        if final_bb > best_bb_score:
            best_bb_score = final_bb
            best_placement = final_placement

    print(f"Best arrangement found with BB-score: {best_bb_score:.4f}")

    assembled = np.zeros((tile_h * grid, tile_w * grid, 3), dtype=pieces[0].dtype)
    for idx, piece_idx in enumerate(best_placement):
        r, c = divmod(idx, grid)
        assembled[r*tile_h:(r+1)*tile_h, c*tile_w:(c+1)*tile_w] = pieces[piece_idx]

    return assembled


def main():













    folders = glob.glob(os.path.join(OUT_IMG, "puzzle_4x4", "*"))

    if not folders:
        print(f"No puzzle folders found in {os.path.join(OUT_IMG, 'puzzle_4x4')}")
        return

    print(f"Found {len(folders)} puzzle folder(s)")
    print("=" * 60)

    success_count = 0
    for folder in folders:
        name = os.path.basename(folder)
        print(f"\nProcessing: {name}")
        print("-" * 60)

        try:
            img = assemble_4x4(folder, strip_width=8, seeds=5, shifter_iters=8)
            out = os.path.join(OUT_ASSEMBLED, f"{name}_assembled.png")
            cv2.imwrite(out, img)
            print(f"[OK] {name} -> {out}")
            success_count += 1
        except Exception as e:
            print(f"[FAIL] {name}: {e}")
            import traceback
            traceback.print_exc()

    print("\n" + "=" * 60)
    print(f"Completed: {success_count}/{len(folders)} puzzles assembled successfully")


if __name__ == "__main__":
    main()
