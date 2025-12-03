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