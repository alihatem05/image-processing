<<<<<<< HEAD
# **README — Image Preprocessing and Tiling Pipeline**

## **1. Overview**

This project implements an automated pipeline that prepares images for later assembly tasks.
The pipeline reads images from a dataset, applies a sequence of preprocessing steps, saves the output of each step, and divides each image into uniform tiles based on the grid size (2×2, 4×4, or 8×8).
The goal is to produce consistent, enhanced, and structured image artifacts that can be used in later reconstruction or analysis stages.

---

## **2. Pipeline Structure**

### **2.1. Dataset Handling**

The system scans the dataset folder and only processes directories that represent a grid size.
Folders named *“correct”* are skipped.
Each image is processed independently.

### **2.2. Preprocessing Stages**

For each image, the following steps are applied:

1. **Grayscale Conversion** – Reduces information and removes color noise.
2. **Gaussian Blur** – Smooths the image and reduces high-frequency noise.
3. **CLAHE** – Increases local contrast and improves feature visibility.
4. **Sharpening Filter** – Highlights edges and small details.
5. **Final Enhancement** – Combines sharpening and smoothing for a balanced result.

Each stage is saved in a dedicated folder under `preprocess/<grid>/<image_name>/`.

### **2.3. Thresholding and Masking**

Otsu thresholding is used to separate foreground and background.
Morphological operations improve the mask by removing small noise and closing gaps.

### **2.4. Image Tiling**

Each image is divided into tiles based on the grid (2×2, 4×4, or 8×8).
Tiles are saved inside:
`final_output/<grid>/<image_name>/`.

---

## **3. Techniques Attempted**

Several preprocessing techniques were tested before selecting the final sequence:

### **3.1. Attempted Techniques**

* **Median Blur**
  Failed to preserve fine texture. Produced overly smooth regions.

* **Bilateral Filter**
  Too slow for large datasets and created inconsistent edge preservation.

* **Adaptive Thresholding**
  Produced unstable masks across images with different lighting.

* **Unsharp Masking Variant**
  Caused halos and made tile borders harder to detect.

* **Histogram Equalization (Global)**
  Introduced strong noise in bright areas and washed out dark regions.

### **3.2. Techniques Selected**

The final pipeline uses:

* Gaussian Blur
* CLAHE
* Sharpening Kernel
* Weighted Sharpening Blend

These were chosen because they gave stable results across the full dataset and kept details sharp without causing artifacts.

---

## **4. Failure Cases Encountered**

During experimentation, the following issues appeared:

* **Noise amplification** when sharpening came before contrast enhancement.
* **Inconsistent threshold masks** on images with heavy shadows.
* **Tile misalignment** when the source image did not divide evenly into the grid.
* **Over-sharpening halos** when using more aggressive kernels.
* **Uneven exposure** causing adaptive thresholding to fail on bright scenes.

These issues informed the order and intensity of the chosen filters.

---

## **5. Suitability for Later Assembly**

The produced artifacts are structured and reliable for downstream assembly tasks:

* **Each preprocessing stage is saved**, allowing later modules to test different combinations.
* **Tiles are uniform and indexed**, which simplifies matching, classification, or feature extraction.
* **Contrast-enhanced outputs** make edges and details easier to detect.
* **Noise reduction + sharpening** results in cleaner boundaries for tile-based assembly.
* **Consistent directory structure** allows automated processing in future phases.

Overall, the output is stable, predictable, and suitable for both machine-learning-based assembly and hand-crafted feature pipelines.

## **6. Important Note**
Since the Dataset folder is too large to push, we removed it. Please add the dataset folder in the same dir as preprossing.py.
=======
# Gravity Falls

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

1. **Grayscale conversion** – simplifies intensity analysis.
2. **Gaussian blur** – suppresses noise while preserving edges.
3. **Laplacian enhancement** – strengthens edge responses.
4. **Uniform grid slicing** – splits the image into equal-sized tiles.

Each input image is divided according to its grid size (2×2, 4×4, or 8×8), and all pieces are saved individually with consistent naming.

Preprocessing outputs are stored for inspection and debugging.

---

## Edge Feature Extraction

For every puzzle piece, the system extracts **four borders** (top, right, bottom, left).

Each border is represented using a multi-channel descriptor composed of:

* LAB color values
* Gradient magnitude (Sobel)
* Gradient direction
* Laplacian response

Borders are normalized to reduce illumination bias and resized to ensure consistent comparison.

---

## Edge Compatibility Measurement

To determine how well two edges fit together, a custom distance metric is used:

* Lp-distance over color, gradient magnitude, gradient direction, and Laplacian channels
* Weighted combination of all components
* Orientation-aware comparison (handles rotated edges)

The result is a **compatibility matrix** that stores the cost of matching any piece edge with another.

---

## Phase 2: Puzzle Assembly

### 2×2 Solver (`2x2.py`)

* Uses **brute-force permutation search** (4! = 24 possibilities)
* Guarantees optimal solution
* Used mainly for validation and benchmarking

### 4×4 Solver (`4x4.py`)

* Uses **best-buddy heuristics**
* Greedy placement with neighborhood constraints
* Segment-based refinement (connected components)
* Optional swap-based local optimization

### 8×8 Solver (`8x8.py`)

* Same strategy as 4×4 with reduced iterations
* Designed to balance accuracy and runtime
* Fully heuristic due to combinatorial explosion

---

## Accuracy Evaluation

**File:** `Compare.py`

The reconstructed image is compared against the ground-truth image:

1. Both images are divided into corresponding tiles.
2. Absolute pixel difference is computed per tile.
3. A similarity score is calculated and thresholded.
4. Accuracy is reported as the percentage of correctly placed tiles.

Thresholds are adjusted based on puzzle size (2×2, 4×4, 8×8).

---

## Visualization

**File:** `visualizations.py`

The visualization module produces:

* Preprocessing step comparisons
* Grid view of extracted pieces
* Final assembled image
* Side-by-side comparison with original image
* Summary figures for all puzzle sizes

These outputs help explain and justify the pipeline decisions.

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

* LAB color space for perceptual consistency
* Gradient-based descriptors for edge continuity
* Best-buddy heuristic for scalable assembly
* No ML models to strictly follow course rules
* Modular design for debugging and evaluation

---

## Limitations

* Assumes rectangular grid-based puzzles
* Performance degrades for highly repetitive textures
* 8×8 assembly is heuristic and not guaranteed optimal

---

## Conclusion

This project demonstrates a complete, end-to-end **classical computer vision solution** for jigsaw puzzle assembly. It integrates preprocessing, feature extraction, geometric reasoning, heuristic optimization, evaluation, and visualization into a cohesive and reproducible system suitable for academic assess
>>>>>>> 7aabc83 (Updating Readme file and Adding the dataset and output)
