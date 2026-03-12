import os
import re
import shutil
import random
import argparse
from typing import Dict, List, Set, Tuple


EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tif", ".tiff"}
SUBJECT_RE = re.compile(r"^([A-Za-z]+)")


def _subject_from_name(filename: str):
    name, _ = os.path.splitext(filename)
    m = SUBJECT_RE.match(name)
    return m.group(1).lower() if m else None


def _list_images(cls_dir: str) -> List[str]:
    out = []
    if not os.path.isdir(cls_dir):
        return out
    for f in os.listdir(cls_dir):
        p = os.path.join(cls_dir, f)
        if os.path.isfile(p) and os.path.splitext(f)[1].lower() in EXTS:
            out.append(f)
    return out


def _split_subjects_all(
    subjects: List[str], ratios: Dict[str, float], seed: int
) -> Dict[str, Set[str]]:
    rng = random.Random(seed)
    order = subjects[:]
    rng.shuffle(order)
    n = len(order)
    n_train = int(ratios["train"] * n)
    n_val = int(ratios["val"] * n)
    train = order[:n_train]
    val = order[n_train:n_train + n_val]
    test = order[n_train + n_val:]
    return {"train": set(train), "val": set(val), "test": set(test)}


def split_person_wise_unified(
    raw_directory: str,
    output_directory: str,
    classes: Tuple[str, str] = ("NotDrowsy", "Drowsy"),
    split_ratios: Dict[str, float] = None,
    seed: int = 42,
    force: bool = False,
) -> Dict[str, Dict[str, int]]:
    """
    Create a person-wise unified split across ALL classes.

    A subject is assigned to exactly one split (train/val/test) globally and
    this assignment is used for every class the subject appears in.

    Directory structure expected:
      raw_directory/
        Drowsy/
        NotDrowsy/

    Output:
      output_directory/{train,val,test}/{Drowsy,NotDrowsy}

    Returns a summary with image counts.
    """
    ratios = split_ratios or {"train": 0.7, "val": 0.15, "test": 0.15}

    # If output exists and has files, skip unless force=True
    if os.path.exists(output_directory):
        has_any_file = False
        for root, _, files in os.walk(output_directory):
            if files:
                has_any_file = True
                break
        if has_any_file and not force:
            # Return existing summary without modifying files
            summary: Dict[str, Dict[str, int]] = {"train": {}, "val": {}, "test": {}}
            for split in ("train", "val", "test"):
                for cls in classes:
                    d = os.path.join(output_directory, split, cls)
                    if os.path.isdir(d):
                        summary[split][cls] = sum(
                            1 for f in os.listdir(d) if os.path.isfile(os.path.join(d, f))
                        )
                    else:
                        summary[split][cls] = 0
            print(f"[INFO] Output exists and non-empty. Skipping split: {output_directory}")
            return summary
        if force:
            shutil.rmtree(output_directory)

    # Prepare destination dirs
    for split in ("train", "val", "test"):
        for cls in classes:
            os.makedirs(os.path.join(output_directory, split, cls), exist_ok=True)

    # Collect unique subjects from all classes
    all_subjects: Set[str] = set()
    class_to_files: Dict[str, List[str]] = {}
    for cls in classes:
        cls_dir = os.path.join(raw_directory, cls)
        files = _list_images(cls_dir)
        class_to_files[cls] = files
        for fname in files:
            subj = _subject_from_name(fname)
            if subj:
                all_subjects.add(subj)

    if not all_subjects:
        raise RuntimeError("No subjects found. Check file naming and directories.")

    # Single unified subject split
    splits = _split_subjects_all(sorted(all_subjects), ratios, seed)

    # Copy images according to the unified subject split
    for cls in classes:
        src_dir = os.path.join(raw_directory, cls)
        for fname in class_to_files[cls]:
            subj = _subject_from_name(fname)
            if subj is None:
                continue
            target_split = (
                "train" if subj in splits["train"] else
                "val"   if subj in splits["val"]   else
                "test"
            )
            dst = os.path.join(output_directory, target_split, cls, fname)
            shutil.copy(os.path.join(src_dir, fname), dst)

    # Summary
    summary: Dict[str, Dict[str, int]] = {"train": {}, "val": {}, "test": {}}
    for split in ("train", "val", "test"):
        for cls in classes:
            d = os.path.join(output_directory, split, cls)
            summary[split][cls] = sum(
                1 for f in os.listdir(d) if os.path.isfile(os.path.join(d, f))
            )
    return summary


def main():
    parser = argparse.ArgumentParser(
        description="Unified person-wise dataset split (train/val/test)"
    )
    parser.add_argument("--raw", required=True, help="Path to raw dataset root")
    parser.add_argument("--out", required=True, help="Path to output split root")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--train", type=float, default=0.7, help="Train ratio")
    parser.add_argument("--val", type=float, default=0.15, help="Val ratio")
    parser.add_argument("--test", type=float, default=0.15, help="Test ratio")
    parser.add_argument("--force", action="store_true", help="Overwrite output directory if exists")

    args = parser.parse_args()

    ratios = {"train": args.train, "val": args.val, "test": args.test}
    if abs(sum(ratios.values()) - 1.0) > 1e-6:
        raise ValueError("Ratios must sum to 1.0")

    print("[INFO] Building unified person-wise split...")
    summary = split_person_wise_unified(
        raw_directory=args.raw,
        output_directory=args.out,
        split_ratios=ratios,
        seed=args.seed,
        force=args.force,
    )

    print("=== Split summary ===")
    for split in ("train", "val", "test"):
        parts = ", ".join(f"{cls}:{summary[split][cls]}" for cls in summary[split])
        print(f"{split:5s} -> {parts}")
    print(f"[DONE] Output written to: {args.out}")


if __name__ == "__main__":
    main()


