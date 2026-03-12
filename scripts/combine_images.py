"""Utility to stitch multiple images into a single output PNG.
Supports both image file paths and matplotlib figures.
"""

from __future__ import annotations

import argparse
import io
from pathlib import Path
from typing import List, Union

import matplotlib.pyplot as plt
from PIL import Image
import os


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Combine multiple images (e.g. GradCAM outputs or training plots) "
            "into a single PNG, either horizontally or vertically."
        )
    )
    parser.add_argument(
        "images",
        nargs="+",
        type=Path,
        help="Paths to the input image files (PNG, JPG, ...).",
    )
    parser.add_argument(
        "--layout",
        choices=("horizontal", "vertical"),
        default="horizontal",
        help="Arrange images side-by-side (horizontal) or stacked (vertical).",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("combined.png"),
        help="Path where the combined image will be saved.",
    )
    return parser.parse_args()


def fig_to_image(fig) -> Image.Image:
    """Convert matplotlib figure to PIL Image."""
    buf = io.BytesIO()
    fig.savefig(buf, format='png', dpi=300, bbox_inches='tight')
    buf.seek(0)
    img = Image.open(buf).convert("RGB")
    buf.close()
    return img


def combine_images(
    inputs: List[Union[Path, plt.Figure]], 
    layout: str
) -> Image.Image:
    """Combine images from file paths or matplotlib figures.
    
    Args:
        inputs: List of image file paths (Path) or matplotlib figures (plt.Figure)
        layout: "horizontal" or "vertical" arrangement
        
    Returns:
        Combined PIL Image
    """
    images = []
    for inp in inputs:
        if isinstance(inp, plt.Figure):
            img = fig_to_image(inp)
            images.append(img)
        elif isinstance(inp, (str, Path)):
            img = Image.open(inp).convert("RGB")
            images.append(img)
        else:
            raise TypeError(f"Unsupported input type: {type(inp)}")

    widths, heights = zip(*(img.size for img in images))

    if layout == "horizontal":
        combined = Image.new("RGB", (sum(widths), max(heights)))
        x_offset = 0
        for img in images:
            combined.paste(img, (x_offset, 0))
            x_offset += img.width
    else:
        combined = Image.new("RGB", (max(widths), sum(heights)))
        y_offset = 0
        for img in images:
            combined.paste(img, (0, y_offset))
            y_offset += img.height

    for img in images:
        img.close()

    return combined


def main() -> None:
    # Simple manual configuration
    PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    layout = "vertical" # "horizontal" or "vertical"

    # Pair A: two GradCAM images (original vs masked) for a single sample
    class_name = "Drowsy"
    subject_prefix = "ZC"
    image_name = f"{subject_prefix}0852.png"
    mask_shape_1 = "original"
    mask_shape_2 = "eyes_mouth"

    masked_inputs = [
        os.path.join(PROJECT_ROOT, "masked_test", mask_shape_1, class_name, f"{subject_prefix}_gradcam", image_name),
        os.path.join(PROJECT_ROOT, "masked_test", mask_shape_2, class_name, f"{subject_prefix}_big_softmax_gradcam", image_name),
    ]
    masked_output = os.path.join(PROJECT_ROOT, "combined_images", class_name, f"{subject_prefix}_gradcam", f"{image_name}_combined.png")

    # Pair B: two training plots from different runs
    metric_name = "training_history" #"test_confusion_matrix"
    metrics_inputs = [
        os.path.join(PROJECT_ROOT, "runs", "30_epoch_without-mask_sbj-gradcam-fixed", "plots", f"{metric_name}.png"),
        os.path.join(PROJECT_ROOT, "runs", "30_epoch_with-mask_sbj-gradcam", "plots", f"{metric_name}.png"),
    ]
    metrics_output = os.path.join(PROJECT_ROOT, "combined_images", "metrics", f"{metric_name}_combined.png")

    def ensure_dir_for(path: str) -> None:
        os.makedirs(os.path.dirname(path), exist_ok=True)

    # Combine pair A
    try:
        ensure_dir_for(masked_output)
        combined_a = combine_images(masked_inputs, layout)
        combined_a.save(masked_output)
        print(f"Saved combined image to {masked_output}")
    except Exception as e:
        print(f"Skipped masked pair: {e}")

    """
    # Combine pair B
    try:
        ensure_dir_for(metrics_output)
        combined_b = combine_images(metrics_inputs, layout)
        combined_b.save(metrics_output)
        print(f"Saved combined image to {metrics_output}")
    except Exception as e:
        print(f"Skipped metrics pair: {e}")
    """

# Example usage with matplotlib figures:
# from scripts.combine_images import combine_images
# fig1, _ = cam.visualize(img1, return_fig=True)
# fig2, _ = cam.visualize(img2, return_fig=True)
# combined = combine_images([fig1, fig2], layout="horizontal")
# combined.save("combined_gradcam.png")
# plt.close(fig1)
# plt.close(fig2)


if __name__ == "__main__":
    main()
