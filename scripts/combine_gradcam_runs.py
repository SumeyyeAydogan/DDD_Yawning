#!/usr/bin/env python
"""
Combine GradCAM outputs and training histories from multiple runs/models,
but with parameters defined manually inside the script (no CLI args).
"""

from __future__ import annotations
from pathlib import Path
from dataclasses import dataclass
from typing import Dict, List, Sequence
from PIL import Image, ImageDraw, ImageFont, ImageColor
import random


@dataclass(frozen=True)
class RunSpec:
    name: str
    path: Path


# ---------------------------------------------------------------------
# 🧩 === MANUAL CONFIGURATION SECTION ===
# ---------------------------------------------------------------------
RUN_SPECS = [
    RunSpec("Original", Path("runs/30_epoch_baseline_e3_yawning/")),
    RunSpec("Reward-soft", Path("runs/30_epoch_reward-soft-e3/")),
    RunSpec("Exp-Reward-soft", Path("runs/30_epoch_exp-reward-soft-e3/")),
    RunSpec("Log-Reward-soft", Path("runs/30_epoch_log-reward-soft-e3/")),
]

GRADCAM_ROOTS = ["plots/val_gradcam", "plots/test_gradcam", "gradcam_custom/"]
OUTPUT_DIR = Path("combined_outputs/soft")
IMAGE_KEYS = None  # ["ZC", "ZA"] , None = process all overlapping keys
MATCH_MODE = "stem"  # "stem" or "filename"
LAYOUT = "vertical"  # "horizontal" or "vertical"
EXTENSIONS = [".png", ".jpg", ".jpeg"]
OUTPUT_SUFFIX = "_combined.png"

FONT_PATH = None  # or Path("path/to/font.ttf")
FONT_SIZE = 30
LABEL_PADDING = 6
LABEL_BG = "#ffffff"
LABEL_FG = "#111111"

SHUFFLE = False
LIMIT = None
SEED = 42

INCLUDE_HISTORY = True
HISTORY_RELPATH = "plots/training_history.png"
HISTORY_OUTPUT_NAME = "training_history_combined.png"
HISTORY_LAYOUT = "vertical"  # or "horizontal"/"vertical"

# ✅ NEW: test confusion matrix merge
INCLUDE_TEST_CONFUSION_MATRIX = True
TEST_CM_RELPATH = "plots/test_confusion_matrix.png"
TEST_CM_OUTPUT_NAME = "test_confusion_matrix_combined.png"
TEST_CM_LAYOUT = "vertical"  # or "horizontal"/"vertical"

VERBOSE = True
# ---------------------------------------------------------------------


def load_font(font_path: Path | None, size: int) -> ImageFont.ImageFont:
    if font_path:
        return ImageFont.truetype(str(font_path), size)
    for candidate in ("DejaVuSans.ttf", "arial.ttf"):
        try:
            return ImageFont.truetype(candidate, size)
        except OSError:
            continue
    return ImageFont.load_default()


def collect_gradcam_index(
    run: RunSpec,
    roots: Sequence[str],
    extensions: Sequence[str],
    match_mode: str,
    verbose: bool = False,
) -> Dict[str, Path]:
    index: Dict[str, Path] = {}
    for rel in roots:
        base = (run.path / rel).expanduser()
        if not base.exists():
            if verbose:
                print(f"[WARN] {run.name}: skipped missing folder {base}")
            continue
        for img_path in base.rglob("*"):
            if img_path.is_file() and img_path.suffix.lower() in extensions:
                key = img_path.stem if match_mode == "stem" else img_path.name
                index.setdefault(key, img_path)
    if verbose:
        print(f"[INFO] {run.name}: indexed {len(index)} GradCAM files")
    return index


def measure_text(draw: ImageDraw.ImageDraw, text: str, font: ImageFont.ImageFont):
    try:
        bbox = draw.textbbox((0, 0), text, font=font)
        return bbox[2] - bbox[0], bbox[3] - bbox[1]
    except AttributeError:
        return draw.textsize(text, font=font)


def create_labeled_panel(image_path: Path, label: str, font, label_bg, label_fg, padding: int):
    with Image.open(image_path) as img:
        base = img.convert("RGB")
        width, height = base.size

    dummy = Image.new("RGB", (10, 10))
    draw = ImageDraw.Draw(dummy)
    text_w, text_h = measure_text(draw, label, font)
    label_height = text_h + padding * 2

    panel = Image.new("RGB", (width, height + label_height), label_bg)
    panel.paste(base, (0, label_height))
    draw = ImageDraw.Draw(panel)
    text_x = max(padding, (width - text_w) // 2)
    draw.text((text_x, padding), label, fill=label_fg, font=font)
    return panel


def stitch_images(images: Sequence[Image.Image], layout: str) -> Image.Image:
    if not images:
        raise ValueError("No images to stitch.")
    widths, heights = zip(*(img.size for img in images))

    if layout == "horizontal":
        combined = Image.new("RGB", (sum(widths), max(heights)), "white")
        x = 0
        for img in images:
            combined.paste(img, (x, 0))
            x += img.width
    else:
        combined = Image.new("RGB", (max(widths), sum(heights)), "white")
        y = 0
        for img in images:
            combined.paste(img, (0, y))
            y += img.height
    return combined


def determine_keys(run_indexes: Dict[RunSpec, Dict[str, Path]], requested_keys, shuffle, limit, seed):
    all_sets = [set(idx.keys()) for idx in run_indexes.values()]
    if not all_sets:
        return []
    if requested_keys:
        available = sorted(set.intersection(*all_sets))
        valid = [k for k in requested_keys if k in available]
        return valid
    keys = sorted(set.intersection(*all_sets))
    if shuffle:
        rng = random.Random(seed)
        rng.shuffle(keys)
    if limit:
        keys = keys[:limit]
    return keys


def combine_gradcam_sets(
    run_specs,
    run_indexes,
    keys,
    layout,
    output_dir,
    output_suffix,
    font,
    label_bg,
    label_fg,
    padding,
):
    output_dir.mkdir(parents=True, exist_ok=True)
    saved_paths: List[Path] = []
    for key in keys:
        panels: List[Image.Image] = []
        for run in run_specs:
            img_path = run_indexes[run].get(key)
            if img_path is None:
                break
            panels.append(create_labeled_panel(img_path, run.name, font, label_bg, label_fg, padding))
        else:
            combined = stitch_images(panels, layout)
            output_path = output_dir / f"{key}{output_suffix}"
            combined.save(output_path)
            combined.close()
            saved_paths.append(output_path)
        for p in panels:
            p.close()
    print(f"[INFO] Saved {len(saved_paths)} combined panels to {output_dir}")
    return saved_paths


def combine_histories(run_specs, history_relpath, output_path, layout, font, label_bg, label_fg, padding):
    panels = []
    for run in run_specs:
        file = run.path / history_relpath
        if file.exists():
            panels.append(create_labeled_panel(file, f"{run.name} history", font, label_bg, label_fg, padding))
        else:
            print(f"[WARN] Missing history file for {run.name}: {file}")
    if len(panels) < 2:
        print("[INFO] Skipping history merge (need at least 2 plots).")
        return None
    combined = stitch_images(panels, layout)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    combined.save(output_path)
    combined.close()
    print(f"[INFO] Saved combined history to {output_path}")
    return output_path


# ✅ NEW: confusion matrix combiner (training_history ile aynı mantık)
def combine_test_confusion_matrices(run_specs, cm_relpath, output_path, layout, font, label_bg, label_fg, padding):
    panels = []
    for run in run_specs:
        file = run.path / cm_relpath
        if file.exists():
            panels.append(
                create_labeled_panel(file, f"{run.name} test CM", font, label_bg, label_fg, padding)
            )
        else:
            print(f"[WARN] Missing test confusion matrix for {run.name}: {file}")

    if len(panels) < 2:
        print("[INFO] Skipping test confusion matrix merge (need at least 2 plots).")
        return None

    combined = stitch_images(panels, layout)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    combined.save(output_path)
    combined.close()
    print(f"[INFO] Saved combined test confusion matrix to {output_path}")
    return output_path


def main():
    font = load_font(FONT_PATH, FONT_SIZE)
    label_bg = ImageColor.getrgb(LABEL_BG)
    label_fg = ImageColor.getrgb(LABEL_FG)
    extensions = [e.lower() if e.startswith(".") else f".{e.lower()}" for e in EXTENSIONS]

    # GradCAM dosyalarını indeksle
    run_indexes: Dict[RunSpec, Dict[str, Path]] = {}
    for run in RUN_SPECS:
        run_indexes[run] = collect_gradcam_index(run, GRADCAM_ROOTS, extensions, MATCH_MODE, VERBOSE)

    keys = determine_keys(run_indexes, IMAGE_KEYS, SHUFFLE, LIMIT, SEED)
    if not keys:
        print("No overlapping GradCAM samples found.")
        return

    combine_gradcam_sets(
        RUN_SPECS,
        run_indexes,
        keys,
        LAYOUT,
        OUTPUT_DIR,
        OUTPUT_SUFFIX,
        font,
        label_bg,
        label_fg,
        LABEL_PADDING,
    )

    if INCLUDE_HISTORY:
        combine_histories(
            RUN_SPECS,
            HISTORY_RELPATH,
            OUTPUT_DIR / HISTORY_OUTPUT_NAME,
            HISTORY_LAYOUT or LAYOUT,
            font,
            label_bg,
            label_fg,
            LABEL_PADDING,
        )

    # ✅ NEW call
    if INCLUDE_TEST_CONFUSION_MATRIX:
        combine_test_confusion_matrices(
            RUN_SPECS,
            TEST_CM_RELPATH,
            OUTPUT_DIR / TEST_CM_OUTPUT_NAME,
            TEST_CM_LAYOUT or LAYOUT,
            font,
            label_bg,
            label_fg,
            LABEL_PADDING,
        )


if __name__ == "__main__":
    main()
