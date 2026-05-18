from __future__ import annotations

from typing import List


def read_manifest(manifest_path: str) -> List[str]:
    files: List[str] = []
    with open(manifest_path, "r", encoding="utf-8") as f:
        for line in f:
            p = line.strip()
            if p:
                files.append(p)
    return files