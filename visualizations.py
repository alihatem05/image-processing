import cv2
import numpy as np
import os
import glob
import re
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec, GridSpecFromSubplotSpec

BASE_OUTPUT = "."
OUT_IMG = os.path.join(BASE_OUTPUT, "final_output")
OUT_ASSEMBLED = os.path.join(BASE_OUTPUT, "assembled")
OUT_PRE = os.path.join(BASE_OUTPUT, "preprocess")
OUT_VIS = os.path.join(BASE_OUTPUT, "visualizations")
os.makedirs(OUT_VIS, exist_ok=True)

_piece_re = re.compile(r"piece[_\-]?(\d+)\.\w+$", re.IGNORECASE)

plt.ion()


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


def visualize_pieces_grid(pieces, grid_size, title="Puzzle Pieces"):
    if not pieces:
        return None

    n = len(pieces)
    cols = grid_size
    rows = (n + cols - 1) // cols

    h, w = pieces[0].shape[:2]

    canvas = np.zeros((rows * h, cols * w, 3), dtype=np.uint8)

    for idx, piece in enumerate(pieces):
        r = idx // cols
        c = idx % cols
        if r < rows and c < cols:
            canvas[r*h:(r+1)*h, c*w:(c+1)*w] = piece

    return canvas


def visualize_pieces_with_labels(pieces, grid_size, title="Puzzle Pieces"):
    if not pieces:
        return None

    n = len(pieces)
    cols = grid_size
    rows = (n + cols - 1) // cols

    h, w = pieces[0].shape[:2]

    label_height = 30
    canvas = np.ones((rows * h + rows * label_height, cols * w, 3), dtype=np.uint8) * 255

    for idx, piece in enumerate(pieces):
        r = idx // cols
        c = idx % cols
        if r < rows and c < cols:
            y_offset = r * label_height
            canvas[r*h+y_offset:(r+1)*h+y_offset, c*w:(c+1)*w] = piece

            label_text = f"Piece {idx}"
            font = cv2.FONT_HERSHEY_SIMPLEX
            font_scale = 0.5
            thickness = 1
            text_size = cv2.getTextSize(label_text, font, font_scale, thickness)[0]
            text_x = c * w + (w - text_size[0]) // 2
            text_y = r * h + y_offset - 5
            cv2.putText(canvas, label_text, (text_x, text_y), font, font_scale, (0, 0, 0), thickness)

    return canvas


def load_preprocessing_steps(image_name, grid_size):
    pre_dir = os.path.join(OUT_PRE, f"{grid_size}x{grid_size}", image_name)
    steps = {}

    if os.path.exists(pre_dir):
        gray_path = os.path.join(pre_dir, "1_gray.png")
        blur_path = os.path.join(pre_dir, "2_blur.png")
        final_path = os.path.join(pre_dir, "3_final.png")

        if os.path.exists(gray_path):
            steps["gray"] = cv2.imread(gray_path)
        if os.path.exists(blur_path):
            steps["blur"] = cv2.imread(blur_path)
        if os.path.exists(final_path):
            steps["final"] = cv2.imread(final_path)

    return steps


def create_comparison_visualization(pieces, assembled_img, image_name, grid_size, preprocessing_steps=None):
    has_preprocessing = preprocessing_steps and len(preprocessing_steps) > 0

    if has_preprocessing:
        fig = plt.figure(figsize=(18, 12))
        gs = GridSpec(3, 2, figure=fig, hspace=0.4, wspace=0.3)
    else:
        fig = plt.figure(figsize=(16, 10))
        gs = GridSpec(2, 2, figure=fig, hspace=0.3, wspace=0.3)

    row_idx = 0

    if has_preprocessing:
        pre_titles = {
            "gray": "1. Grayscale",
            "blur": "2. Gaussian Blur",
            "final": "3. Final Enhanced"
        }
        ax_pre = fig.add_subplot(gs[row_idx, :])
        ax_pre.axis('off')

        gs_pre = GridSpecFromSubplotSpec(1, 3, subplot_spec=gs[row_idx, :], wspace=0.3)

        step_names = ["gray", "blur", "final"]
        for idx, step_name in enumerate(step_names):
            if step_name in preprocessing_steps:
                ax = fig.add_subplot(gs_pre[0, idx])
                step_img = preprocessing_steps[step_name]
                if len(step_img.shape) == 2:
                    ax.imshow(step_img, cmap='gray')
                else:
                    step_rgb = cv2.cvtColor(step_img, cv2.COLOR_BGR2RGB)
                    ax.imshow(step_rgb)
                ax.set_title(pre_titles[step_name], fontsize=12, fontweight='bold')
                ax.axis('off')
        row_idx += 1

    ax1 = fig.add_subplot(gs[row_idx, :])
    pieces_grid = visualize_pieces_grid(pieces, grid_size)
    if pieces_grid is not None:
        pieces_rgb = cv2.cvtColor(pieces_grid, cv2.COLOR_BGR2RGB)
        ax1.imshow(pieces_rgb)
        ax1.set_title(f"Preprocessed Pieces ({len(pieces)} pieces)", fontsize=14, fontweight='bold')
        ax1.axis('off')
    row_idx += 1

    if has_preprocessing:
        ax2 = fig.add_subplot(gs[row_idx, 0])
        if assembled_img is not None:
            assembled_rgb = cv2.cvtColor(assembled_img, cv2.COLOR_BGR2RGB)
            ax2.imshow(assembled_rgb)
            ax2.set_title("Assembled Image (After Matching)", fontsize=14, fontweight='bold')
            ax2.axis('off')

        ax4 = fig.add_subplot(gs[row_idx, 1])
    else:
        ax2 = fig.add_subplot(gs[row_idx, 0])
        if assembled_img is not None:
            assembled_rgb = cv2.cvtColor(assembled_img, cv2.COLOR_BGR2RGB)
            ax2.imshow(assembled_rgb)
            ax2.set_title("Assembled Image (After Matching)", fontsize=14, fontweight='bold')
            ax2.axis('off')

        ax4 = fig.add_subplot(gs[row_idx, 1])

    dataset_path = os.path.join(BASE_OUTPUT, "dataset", "Gravity Falls")
    original_img = None
    if os.path.exists(dataset_path):
        for folder in os.listdir(dataset_path):
            if f"{grid_size}x{grid_size}" in folder.lower():
                folder_path = os.path.join(dataset_path, folder)
                if os.path.isdir(folder_path):
                    for ext in ["png", "jpg", "jpeg"]:
                        orig_path = os.path.join(folder_path, f"{image_name}.{ext}")
                        if os.path.exists(orig_path):
                            original_img = cv2.imread(orig_path)
                            break
                    if original_img is not None:
                        break
                if original_img is not None:
                    break

    if original_img is not None:
        original_rgb = cv2.cvtColor(original_img, cv2.COLOR_BGR2RGB)
        ax4.imshow(original_rgb)
        ax4.set_title("Original Image", fontsize=14, fontweight='bold')
        ax4.axis('off')
    else:
        ax4.axis('off')
        ax4.text(0.5, 0.5, "Original\nNot Found", ha='center', va='center', 
                fontsize=12, transform=ax4.transAxes)

    plt.suptitle(f"Visualization: {image_name} ({grid_size}x{grid_size})", 
                 fontsize=18, fontweight='bold', y=0.995)

    return fig


def visualize_single_puzzle(image_name, grid_type, grid_size, show_popup=True):
    piece_dir = None
    for folder in os.listdir(OUT_IMG):
        if grid_type.lower() in folder.lower():
            folder_path = os.path.join(OUT_IMG, folder)
            if os.path.isdir(folder_path):
                image_dir = os.path.join(folder_path, image_name)
                if os.path.isdir(image_dir):
                    piece_dir = image_dir
                    break

    if piece_dir is None:
        print(f"[WARN] Pieces directory not found for {image_name} ({grid_type})")
        return None

    pieces = read_pieces_sorted(piece_dir)
    if not pieces:
        print(f"[WARN] No pieces found for {image_name} ({grid_type})")
        return None

    assembled_path = os.path.join(OUT_ASSEMBLED, grid_type, f"{image_name}_assembled.png")
    assembled_img = None
    if os.path.exists(assembled_path):
        assembled_img = cv2.imread(assembled_path)

    preprocessing_steps = load_preprocessing_steps(image_name, grid_size)

    fig = create_comparison_visualization(pieces, assembled_img, image_name, grid_size, preprocessing_steps)

    if show_popup:
        plt.show()
        input(f"Press Enter to continue to next visualization...")

    vis_dir = os.path.join(OUT_VIS, grid_type)
    os.makedirs(vis_dir, exist_ok=True)
    output_path = os.path.join(vis_dir, f"{image_name}_visualization.png")
    fig.savefig(output_path, dpi=150, bbox_inches='tight')

    if not show_popup:
        plt.close(fig)

    print(f"[INFO] Saved visualization: {output_path}")
    return output_path


def visualize_all_puzzles(show_popup=True, max_per_grid=4):
    results = []

    for grid_type in ["2x2", "4x4", "8x8"]:
        grid_size = int(grid_type[0])
        print(f"\n{'='*60}")
        print(f"Processing {grid_type} puzzles")
        print(f"{'='*60}")

        puzzle_folders = []
        for folder in os.listdir(OUT_IMG):
            if grid_type.lower() in folder.lower():
                folder_path = os.path.join(OUT_IMG, folder)
                if os.path.isdir(folder_path):
                    puzzle_folders.append(folder_path)

        if not puzzle_folders:
            print(f"[WARN] No folders found for {grid_type}")
            continue

        all_images = []
        for folder_path in puzzle_folders:
            for image_name in sorted(os.listdir(folder_path)):
                image_dir = os.path.join(folder_path, image_name)
                if os.path.isdir(image_dir):
                    all_images.append((folder_path, image_name))

        images_to_process = all_images[:max_per_grid]
        print(f"[INFO] Showing first {len(images_to_process)} images for {grid_type}")

        for folder_path, image_name in images_to_process:
            try:
                output_path = visualize_single_puzzle(image_name, grid_type, grid_size, show_popup=show_popup)
                if output_path:
                    results.append({
                        "image": image_name,
                        "grid": grid_type,
                        "visualization": output_path
                    })
            except Exception as e:
                print(f"[ERROR] Failed to visualize {image_name} ({grid_type}): {e}")
                import traceback
                traceback.print_exc()

        if len(all_images) > max_per_grid:
            remaining_count = len(all_images) - max_per_grid
            print(f"\n[INFO] Saving remaining {remaining_count} images for {grid_type} (writing to terminal)...")
            for idx, (folder_path, image_name) in enumerate(all_images[max_per_grid:], 1):
                try:
                    print(f"  [{idx}/{remaining_count}] Processing: {image_name}...", end=" ", flush=True)
                    output_path = visualize_single_puzzle(image_name, grid_type, grid_size, show_popup=False)
                    if output_path:
                        results.append({
                            "image": image_name,
                            "grid": grid_type,
                            "visualization": output_path
                        })
                        print("✓ Saved")
                    else:
                        print("✗ Failed")
                except Exception as e:
                    print(f"✗ Error: {e}")

    return results


def create_summary_visualization():
    fig, axes = plt.subplots(3, 3, figsize=(18, 18))
    fig.suptitle("Summary: Preprocessed Pieces and Assembled Images", 
                 fontsize=18, fontweight='bold', y=0.995)

    grid_types = ["2x2", "4x4", "8x8"]

    for row, grid_type in enumerate(grid_types):
        grid_size = int(grid_type[0])

        found = False
        for folder in os.listdir(OUT_IMG):
            if grid_type.lower() in folder.lower():
                folder_path = os.path.join(OUT_IMG, folder)
                if os.path.isdir(folder_path):
                    for image_name in sorted(os.listdir(folder_path))[:1]:
                        image_dir = os.path.join(folder_path, image_name)
                        if os.path.isdir(image_dir):
                            pieces = read_pieces_sorted(image_dir)
                            if pieces:
                                pieces_grid = visualize_pieces_grid(pieces, grid_size)
                                if pieces_grid is not None:
                                    pieces_rgb = cv2.cvtColor(pieces_grid, cv2.COLOR_BGR2RGB)
                                    axes[row, 0].imshow(pieces_rgb)
                                    axes[row, 0].set_title(f"{grid_type} - Preprocessed Pieces", 
                                                           fontsize=12, fontweight='bold')
                                    axes[row, 0].axis('off')

                                assembled_path = os.path.join(OUT_ASSEMBLED, grid_type, 
                                                             f"{image_name}_assembled.png")
                                if os.path.exists(assembled_path):
                                    assembled_img = cv2.imread(assembled_path)
                                    assembled_rgb = cv2.cvtColor(assembled_img, cv2.COLOR_BGR2RGB)
                                    axes[row, 1].imshow(assembled_rgb)
                                    axes[row, 1].set_title(f"{grid_type} - Assembled", 
                                                          fontsize=12, fontweight='bold')
                                    axes[row, 1].axis('off')

                                if len(pieces) <= 16:
                                    pieces_labeled = visualize_pieces_with_labels(pieces, grid_size)
                                    if pieces_labeled is not None:
                                        pieces_labeled_rgb = cv2.cvtColor(pieces_labeled, cv2.COLOR_BGR2RGB)
                                        axes[row, 2].imshow(pieces_labeled_rgb)
                                        axes[row, 2].set_title(f"{grid_type} - Labeled Pieces", 
                                                              fontsize=12, fontweight='bold')
                                        axes[row, 2].axis('off')

                                found = True
                                break
                    if found:
                        break
                if found:
                    break

    plt.tight_layout()
    summary_path = os.path.join(OUT_VIS, "summary_visualization.png")
    fig.savefig(summary_path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"\n[INFO] Summary visualization saved: {summary_path}")
    return summary_path


def main():
    print("="*60)
    print("Puzzle Visualization Generator")
    print("Showing first 4 images per puzzle type as pop-ups")
    print("="*60)

    print("\n[1/2] Creating individual puzzle visualizations with pop-ups...")
    print("Note: First 4 images per grid type will be shown as pop-ups")
    print("Press Enter after each visualization to continue...\n")
    results = visualize_all_puzzles(show_popup=True, max_per_grid=4)

    print("\n[2/2] Creating summary visualization...")
    create_summary_visualization()

    print("\n" + "="*60)
    print("VISUALIZATION SUMMARY")
    print("="*60)
    print(f"Total visualizations created: {len(results)}")

    by_grid = {}
    for r in results:
        grid = r["grid"]
        by_grid[grid] = by_grid.get(grid, 0) + 1

    for grid, count in sorted(by_grid.items()):
        print(f"  {grid}: {count} visualizations")

    print(f"\nOutput directory: {OUT_VIS}")
    print("="*60)

    print("\nAll visualizations complete! Close the plot windows to exit.")
    plt.ioff()
    plt.show()


if __name__ == "__main__":
    main()

