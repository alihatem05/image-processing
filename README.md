# Gravity Falls Puzzle

## Course Information

* **Course**: CSE483 / CESS5004 – Computer Vision
* **Faculty**: Engineering, Ain Shams University
* **Semester**: Fall 2025
* **Project Type**: Course Project (Milestone 1 & Milestone 2)

---

## Project Overview

This project implements a **fully classical computer vision pipeline** to automatically process, analyze, and assemble jigsaw puzzle images. The system takes complete puzzle images, preprocesses them, segments them into tiles, extracts edge features, matches neighboring pieces, and reconstructs the original image layout.

No machine learning or deep learning techniques are used. All steps rely strictly on **image processing, geometry, and heuristic optimization**, in compliance with course rules.

The project supports puzzles of sizes:

* **2×2**
* **4×4**
* **8×8**

---

## Objectives

1. Design a robust preprocessing pipeline to enhance puzzle edges.
2. Segment puzzle images into individual pieces.
3. Extract edge descriptors for each puzzle piece.
4. Compute compatibility scores between neighboring edges.
5. Assemble puzzles automatically using edge matching and heuristic search.
6. Evaluate reconstruction accuracy against ground-truth images.
7. Visualize intermediate and final results clearly.

---

## Project Structure

```
Gravity Falls/
│
├── dataset/
│   └── Gravity Falls/
│       ├── puzzle_2x2/
│       ├── puzzle_4x4/
│       ├── puzzle_8x8/
│       └── correct/
│
├── final_output/          # Segmented puzzle pieces (Phase 1 output)
│   ├── puzzle_2x2/
│   ├── puzzle_4x4/
│   └── puzzle_8x8/
│
├── preprocess/            # Saved preprocessing steps
│   ├── 2x2/
│   ├── 4x4/
│   └── 8x8/
│
├── assembled/             # Final assembled images
│   ├── 2x2/
│   ├── 4x4/
│   └── 8x8/
│
├── visualizations/        # Visual comparison figures
│
├── Phase1.py              # Preprocessing & segmentation
├── 2x2.py                 # 2×2 puzzle solver (brute force)
├── 4x4.py                 # 4×4 puzzle solver (heuristic)
├── 8x8.py                 # 8×8 puzzle solver (heuristic)
├── Compare.py             # Accuracy evaluation
├── visualizations.py      # Visualization and reporting
└── README.md
```

---

## Phase 1: Preprocessing & Segmentation

**File:** `Phase1.py`

### Steps Performed

1. **Grayscale conversion** – converts BGR to grayscale for simplified intensity analysis.
2. **Gaussian blur** – applies 3×3 Gaussian blur to suppress noise while preserving edges.
3. **Laplacian enhancement** – computes Laplacian operator (ksize=3) to detect edge responses.
4. **Weighted combination** – combines blurred image (85%) with Laplacian-enhanced image (15%) to strengthen edge features.
5. **Uniform grid slicing** – splits the original image into equal-sized tiles based on grid dimensions.

Each input image is divided according to its grid size (2×2, 4×4, or 8×8), and all pieces are saved individually as `piece_1.png`, `piece_2.png`, etc.

Preprocessing intermediate steps (grayscale, blur, final enhanced) are saved in `preprocess/{grid_size}/{image_name}/` for inspection and debugging.

---

## Edge Feature Extraction

For every puzzle piece, the system extracts **four borders** (top=0, right=1, bottom=2, left=3) using a configurable strip width (default: 8 pixels).

Each border is represented using a multi-channel descriptor composed of:

* **LAB color values** (3 channels) – converted from BGR for perceptual consistency
* **Gradient magnitude** (1 channel) – computed using Sobel operators (3×3 kernel)
* **Gradient direction** (1 channel) – phase angle in degrees from Sobel gradients
* **Laplacian response** (1 channel) – second-order derivative for edge detection

The extraction process:
1. Converts piece to LAB color space
2. Applies Gaussian blur (3×3) to the grayscale version
3. Computes Sobel gradients (Gx, Gy) and derives magnitude and direction
4. Computes Laplacian operator
5. Concatenates all channels into a 6-channel descriptor

Borders are normalized (zero mean, unit variance per channel) to reduce illumination bias and resized to ensure consistent comparison when dimensions differ.

---

## Edge Compatibility Measurement

To determine how well two edges fit together, a custom distance metric is used:

**Distance Function Parameters:**
* **p = 0.3** – Lp-norm exponent for component-wise differences
* **q = 1/16** – Final distance normalization exponent
* **Weights:**
  * Color (LAB): 0.4
  * Gradient magnitude: 0.2
  * Gradient direction: 0.2
  * Laplacian: 0.2

**Process:**
1. Borders are oriented (transposed for vertical edges) to align for comparison
2. Normalized strips are resized if dimensions differ
3. Component-wise Lp-distances are computed for each channel type
4. Weighted sum is calculated and raised to the power (q/p) for final distance

The result is a **compatibility matrix** (4 dictionaries, one per side) that stores the cost of matching any piece edge with another. Lower values indicate better compatibility.

---

## Phase 2: Puzzle Assembly

### 2×2 Solver (`2x2.py`)

* Uses **brute-force permutation search** (4! = 24 possibilities)
* Evaluates all permutations and selects the one with minimum total compatibility cost
* Guarantees optimal solution
* Used mainly for validation and benchmarking

### 4×4 Solver (`4x4.py`)

* Uses **best-buddy heuristics** with greedy placement
* **Placement Strategy:**
  1. Starts with a random seed piece at a random position
  2. Prioritizes empty slots with the most filled neighbors
  3. For each slot, evaluates candidates based on:
     - Best-buddy count (mutual best matches with neighbors)
     - Average compatibility score with neighbors
  4. Selects piece with highest best-buddy count, breaking ties by lowest compatibility
* **Refinement:**
  - Segment-based refinement using connected components analysis
  - Optional swap-based local optimization to improve placement
* Multiple seed placements are tried, and the best result is selected

### 8×8 Solver (`8x8.py`)

* Same strategy as 4×4 with optimized parameters
* Uses fewer seed placements and iterations to balance accuracy and runtime
* Fully heuristic due to combinatorial explosion (64! possibilities)
* Designed for scalability while maintaining reasonable accuracy

---

## Accuracy Evaluation

**File:** `Compare.py`

The reconstructed image is compared against the ground-truth image from the `correct/` folder:

1. Both images are divided into corresponding tiles based on grid dimensions.
2. For each tile pair:
   - Absolute pixel difference is computed using `cv2.absdiff()`
   - Similarity score = 1 - (sum of differences / (tile_size × 255))
3. A tile is considered correct if its similarity score exceeds the threshold.
4. Accuracy is reported as the percentage of correctly placed tiles.

**Thresholds by puzzle size:**
* **2×2**: 0.6 (60% similarity required)
* **4×4**: 0.7 (70% similarity required)
* **8×8**: 0.75 (75% similarity required)

The evaluation reports per-image accuracy and average accuracy per puzzle size.

---

## Visualization

**File:** `visualizations.py`

The visualization module produces comprehensive visual outputs:

**Per-Image Visualizations:**
* Preprocessing step comparisons (grayscale, blur, final enhanced) in a horizontal row
* Grid view of all extracted puzzle pieces
* Final assembled image (after matching and placement)
* Side-by-side comparison with the original image from the dataset

**Summary Visualizations:**
* Summary figure showing one example from each puzzle size (2×2, 4×4, 8×8)
* Displays preprocessed pieces, assembled result, and labeled pieces (for ≤16 pieces)

**Features:**
* First 4 images per grid type are shown as interactive pop-ups (press Enter to continue)
* Remaining images are processed silently and saved
* All visualizations are saved to `visualizations/{grid_type}/{image_name}_visualization.png`
* Summary saved to `visualizations/summary_visualization.png`

These outputs help explain and justify the pipeline decisions and demonstrate the system's performance.

---

## How to Run

1. **Run preprocessing and segmentation**

   ```bash
   python Phase1.py
   ```

2. **Assemble puzzles**

   ```bash
   python 2x2.py
   python 4x4.py
   python 8x8.py
   ```

3. **Evaluate accuracy**

   ```bash
   python Compare.py
   ```

4. **Generate visualizations**

   ```bash
   python visualizations.py
   ```

---

## Design Choices & Justification

* **LAB color space** – Perceptually uniform color representation, reducing illumination bias in edge matching
* **Multi-channel edge descriptors** – Combines color, gradient magnitude, gradient direction, and Laplacian to capture both appearance and structural edge information
* **Weighted Lp-distance** – Balances different feature types (40% color, 20% each for gradients and Laplacian) for robust matching
* **Best-buddy heuristic** – Scalable approach that prioritizes mutual best matches, enabling efficient assembly for larger puzzles
* **Greedy placement with neighborhood constraints** – Ensures local consistency while building the global solution
* **Segment-based refinement** – Uses connected components to identify and fix misaligned regions
* **No ML models** – Strictly classical computer vision approach to comply with course requirements
* **Modular design** – Separate files for preprocessing, solving, evaluation, and visualization enable easy debugging and evaluation

---

## Limitations

* Assumes rectangular grid-based puzzles
* Performance degrades for highly repetitive textures
* 8×8 assembly is heuristic and not guaranteed optimal

---

## Conclusion

This project demonstrates a complete, end-to-end **classical computer vision solution** for jigsaw puzzle assembly. It integrates preprocessing, feature extraction, geometric reasoning, heuristic optimization, evaluation, and visualization into a cohesive and reproducible system suitable for academic assessment.

The system successfully handles puzzles of varying complexity (2×2, 4×4, 8×8) using a unified approach that scales from brute-force optimal solutions to efficient heuristic methods, all while maintaining strict adherence to classical computer vision techniques without any machine learning components.
