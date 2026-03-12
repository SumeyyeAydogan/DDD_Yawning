# split_new_dataset_fixed.py
import os, re, shutil, random, csv
from collections import defaultdict
from typing import Optional, Dict, List, Set, Tuple

EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tif", ".tiff"}
SUBJECT_RE = re.compile(r"^([A-Za-z]+)")

def _subject_from_name(filename: str) -> Optional[str]:
    name, _ = os.path.splitext(filename)
    m = SUBJECT_RE.match(name)
    if not m:
        return None
    return m.group(1).lower()

def _list_images(cls_dir: str):
    for f in os.listdir(cls_dir):
        p = os.path.join(cls_dir, f)
        if os.path.isfile(p) and os.path.splitext(f)[1].lower() in EXTS:
            yield f

def split_dataset(
    raw_directory: str,
    output_directory: str,
    classes: Tuple[str, str] = ("NotDrowsy", "Drowsy"),
    split_ratios: Optional[Dict[str, float]] = None,
    seed: int = 42,
    verbose: bool = True,
):
    """
    Per-subject stratified split into train/val/test.
    Each subject appears in all splits; for each subject and each class, images
    are partitioned by the given ratios (e.g., 0.7/0.15/0.15), preserving the
    within-subject class distribution across splits.
    """
    ratios = split_ratios or {"train": 0.7, "val": 0.15, "test": 0.15}

    # Reset destination dir if exists
    if os.path.isdir(output_directory):
        if verbose:
            print(f"[INFO] Removing existing output directory: {output_directory}")
        shutil.rmtree(output_directory)

    # Prepare destination dirs
    for split in ("train", "val", "test"):
        for cls in classes:
            os.makedirs(os.path.join(output_directory, split, cls), exist_ok=True)

    # 1. COLLECT IMAGES PER SUBJECT AND CLASS
    #    subject_to_files[cls][subject] = [filenames]
    subject_to_files: Dict[str, Dict[str, List[str]]] = {cls: {} for cls in classes}
    for cls in classes:
        cls_src = os.path.join(raw_directory, cls)
        if not os.path.isdir(cls_src):
            raise FileNotFoundError(f"Class directory not found: {cls_src}")

        for fname in _list_images(cls_src):
            subj = _subject_from_name(fname)
            if subj is None:
                if verbose:
                    print(f"[WARN] Subject not parsed, skipping: {fname}")
                continue
            subject_to_files[cls].setdefault(subj, []).append(fname)

    # 2. PER-SUBJECT, PER-CLASS SPLIT BY RATIOS
    rng = random.Random(seed)
    summary: Dict[str, Dict[str, int]] = {"train": {cls: 0 for cls in classes},
                                          "val":   {cls: 0 for cls in classes},
                                          "test":  {cls: 0 for cls in classes}}
    # Per-subject, per-class detail summary
    subject_detail: Dict[str, Dict[str, Dict[str, int]]] = {cls: {} for cls in classes}

    def split_counts(n: int, r_train: float, r_val: float, r_test: float) -> Tuple[int, int, int]:
        # Use floor for first two; remainder goes to test to ensure total matches
        n_train = int(n * r_train)
        n_val = int(n * r_val)
        n_test = max(0, n - (n_train + n_val))
        return n_train, n_val, n_test

    for cls in classes:
        cls_src = os.path.join(raw_directory, cls)
        for subj, files in subject_to_files[cls].items():
            rng.shuffle(files)
            n = len(files)
            t_count, v_count, te_count = split_counts(n, ratios["train"], ratios["val"], ratios["test"])

            # Assign files to splits
            train_files = files[:t_count]
            val_files   = files[t_count:t_count+v_count]
            test_files  = files[t_count+v_count:]

            # init subject detail buckets
            subject_detail[cls].setdefault(subj, {"train": 0, "val": 0, "test": 0, "total": n})

            for split_name, split_files in [("train", train_files), ("val", val_files), ("test", test_files)]:
                dst_dir = os.path.join(output_directory, split_name, cls)
                for fname in split_files:
                    src_path = os.path.join(cls_src, fname)
                    dst_path = os.path.join(dst_dir, fname)
                    shutil.copy2(src_path, dst_path)
                summary[split_name][cls] += len(split_files)
                subject_detail[cls][subj][split_name] += len(split_files)

    # Write per-subject detailed CSV
    csv_path = os.path.join(output_directory, "subject_split_summary.csv")
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["subject", "class", "train", "val", "test", "total", "r_train", "r_val", "r_test"])
        for cls in classes:
            for subj, counts in sorted(subject_detail[cls].items()):
                total = max(1, counts.get("total", 0))
                r_tr = counts["train"] / total
                r_va = counts["val"] / total
                r_te = counts["test"] / total
                writer.writerow([subj, cls, counts["train"], counts["val"], counts["test"], total,
                                 f"{r_tr:.3f}", f"{r_va:.3f}", f"{r_te:.3f}"])

    if verbose:
        print("=== Per-subject stratified split summary ===")
        for split in ("train", "val", "test"):
            print(f"{split:5s} -> " + ", ".join(f"{cls}:{summary[split][cls]}" for cls in classes))
        print(f"[INFO] Subject-level summary written to: {csv_path}")
    
    return summary

if __name__ == "__main__":
    # Test
    raw_dir = "new_dataset"
    output_dir = "new_splitted_dataset_fixed"
    
    summary = split_new_dataset_fixed(
        raw_directory=raw_dir,
        output_directory=output_dir,
        classes=("NotDrowsy", "Drowsy"),  # Duzeltildi
        seed=42,
        verbose=True
    )
    
    print(f"\n✅ Fixed dataset created: {output_dir}")
    print("📊 Summary:", summary)
