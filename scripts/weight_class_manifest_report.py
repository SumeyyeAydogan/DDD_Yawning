import argparse
import csv
import json
import os
from pathlib import Path
from statistics import mean, median
from typing import Dict, List, Optional, Tuple


def _infer_class_from_rel(rel_path: str) -> str:
    p = rel_path.replace("\\", "/").lstrip("./")
    if p.startswith("NoYawn/") or "/NoYawn/" in p:
        return "NoYawn"
    if p.startswith("Yawn/") or "/Yawn/" in p:
        return "Yawn"
    return "Other"


def _q(values: List[float], q: float) -> float:
    if not values:
        return float("nan")
    v = sorted(values)
    idx = int(q * (len(v) - 1))
    return float(v[idx])


def _stats(values: List[float]) -> Dict[str, float]:
    if not values:
        return {
            "n": 0,
            "mean": float("nan"),
            "median": float("nan"),
            "q25": float("nan"),
            "q75": float("nan"),
            "min": float("nan"),
            "max": float("nan"),
        }
    return {
        "n": int(len(values)),
        "mean": float(mean(values)),
        "median": float(median(values)),
        "q25": _q(values, 0.25),
        "q75": _q(values, 0.75),
        "min": float(min(values)),
        "max": float(max(values)),
    }


def _read_manifest_relpaths(manifest_path: Path, data_dir: Path) -> List[str]:
    rels: List[str] = []
    with open(manifest_path, "r", encoding="utf-8") as f:
        for line in f:
            p = line.strip()
            if not p:
                continue
            abs_p = Path(p)
            if abs_p.is_absolute():
                rel = os.path.relpath(str(abs_p), str(data_dir)).replace("\\", "/")
            else:
                rel = p.replace("\\", "/").lstrip("./")
            rels.append(rel)
    return rels


def _load_weights(weights_json_path: Path) -> Dict[str, float]:
    with open(weights_json_path, "r", encoding="utf-8") as f:
        raw = json.load(f)
    return {str(k).replace("\\", "/").lstrip("./"): float(v) for k, v in raw.items()}


def _fold_id_from_name(name: str) -> Optional[int]:
    # expected: fold_3_weights
    parts = name.split("_")
    if len(parts) < 2:
        return None
    try:
        return int(parts[1])
    except ValueError:
        return None


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Class-wise report for optimized GradCAM weights (optionally restricted by fold manifests)."
    )
    parser.add_argument("--weights-root", required=True, help="Path like artifacts/.../_tmp_auto_opt")
    parser.add_argument("--data-dir", required=True, help="Dataset root used by weights (e.g. ydd_splitted_dataset/train)")
    parser.add_argument(
        "--manifest-root",
        default=None,
        help="Optional folder containing fold manifests (e.g. runs/.../cv_logs/no_weights/manifests)",
    )
    parser.add_argument(
        "--manifest-split",
        default="train",
        choices=["train", "val"],
        help="Manifest file suffix to use when manifest-root is provided.",
    )
    parser.add_argument(
        "--output-dir",
        default=None,
        help="Output directory (default: <weights-root>/class_weight_report)",
    )
    args = parser.parse_args()

    weights_root = Path(args.weights_root)
    data_dir = Path(args.data_dir)
    manifest_root = Path(args.manifest_root) if args.manifest_root else None
    output_dir = Path(args.output_dir) if args.output_dir else (weights_root / "class_weight_report")
    output_dir.mkdir(parents=True, exist_ok=True)

    rows: List[Dict[str, object]] = []
    fold_dirs = sorted([d for d in weights_root.iterdir() if d.is_dir() and d.name.startswith("fold_")])
    if not fold_dirs:
        raise SystemExit(f"No fold_* directories found under: {weights_root}")

    for fd in fold_dirs:
        fold_id = _fold_id_from_name(fd.name)
        if fold_id is None:
            continue

        weights_json = fd / "optimized_gradcam_weights.json"
        if not weights_json.exists():
            continue

        weights_map = _load_weights(weights_json)
        selected_keys = list(weights_map.keys())

        if manifest_root is not None:
            manifest_path = manifest_root / f"fold_{fold_id}_{args.manifest_split}_files.txt"
            if manifest_path.exists():
                rels = _read_manifest_relpaths(manifest_path, data_dir)
                rel_set = set(rels)
                selected_keys = [k for k in selected_keys if k in rel_set]

        vals_by_class: Dict[str, List[float]] = {"NoYawn": [], "Yawn": [], "Other": []}
        for rel in selected_keys:
            cls = _infer_class_from_rel(rel)
            vals_by_class.setdefault(cls, []).append(weights_map[rel])

        no_stats = _stats(vals_by_class.get("NoYawn", []))
        y_stats = _stats(vals_by_class.get("Yawn", []))
        diff = float(y_stats["mean"] - no_stats["mean"]) if (no_stats["n"] > 0 and y_stats["n"] > 0) else float("nan")

        rows.append(
            {
                "fold_id": int(fold_id),
                "scope": "manifest" if manifest_root is not None else "all_weights",
                "n_total_selected": int(len(selected_keys)),
                "NoYawn_n": int(no_stats["n"]),
                "NoYawn_mean": float(no_stats["mean"]),
                "NoYawn_median": float(no_stats["median"]),
                "Yawn_n": int(y_stats["n"]),
                "Yawn_mean": float(y_stats["mean"]),
                "Yawn_median": float(y_stats["median"]),
                "mean_diff_Yawn_minus_NoYawn": diff,
            }
        )

    if not rows:
        raise SystemExit("No rows produced; check weights-root path and files.")

    # overall aggregate
    agg_no = [float(r["NoYawn_mean"]) for r in rows if r["NoYawn_n"] > 0]
    agg_y = [float(r["Yawn_mean"]) for r in rows if r["Yawn_n"] > 0]
    aggregate = {
        "fold_count": int(len(rows)),
        "scope": rows[0]["scope"],
        "mean_of_fold_NoYawn_mean": float(mean(agg_no)) if agg_no else float("nan"),
        "mean_of_fold_Yawn_mean": float(mean(agg_y)) if agg_y else float("nan"),
        "mean_diff_Yawn_minus_NoYawn": float(mean(agg_y) - mean(agg_no)) if (agg_no and agg_y) else float("nan"),
        "rows": rows,
    }

    out_json = output_dir / "class_weight_report.json"
    out_csv = output_dir / "class_weight_report.csv"
    with open(out_json, "w", encoding="utf-8") as f:
        json.dump(aggregate, f, indent=2, ensure_ascii=False)

    with open(out_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "fold_id",
                "scope",
                "n_total_selected",
                "NoYawn_n",
                "NoYawn_mean",
                "NoYawn_median",
                "Yawn_n",
                "Yawn_mean",
                "Yawn_median",
                "mean_diff_Yawn_minus_NoYawn",
            ],
        )
        writer.writeheader()
        for r in rows:
            writer.writerow(r)

    print(f"[DONE] JSON: {out_json}")
    print(f"[DONE] CSV : {out_csv}")
    print(
        "[SUMMARY] mean_diff_Yawn_minus_NoYawn = "
        f"{aggregate['mean_diff_Yawn_minus_NoYawn']:+.4f} "
        "(positive => Yawn gets higher weights)"
    )


if __name__ == "__main__":
    main()

