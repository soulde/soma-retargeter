#!/usr/bin/env python3
"""Select mirrored, category-balanced SOMA locomotion BVHs from a ZIP archive."""

import argparse
import json
import re
import zipfile
from collections import defaultdict
from pathlib import Path, PurePosixPath


DEFAULT_QUOTAS = {
    "walk_forward": 32,
    "jog_forward": 32,
    "walk_backward": 24,
    "jog_backward": 24,
    "walk_left": 16,
    "walk_right": 16,
    "walk_diagonal": 16,
    "jog_left": 16,
    "jog_right": 16,
    "jog_diagonal": 16,
    "turn_walk": 24,
    "turn_jog": 24,
    "stairs": 16,
}


def _gait(name):
    return "walk" if "walk" in name else "jog"


def classify_motion(path):
    """Classify a motion by filename into a balanced locomotion bucket."""
    name = PurePosixPath(path).name.lower()
    if "stair" in name:
        return "stairs"
    if "sideway" in name:
        gait = _gait(name)
        if "sideway_left" in name:
            return f"{gait}_left"
        if "sideway_right" in name:
            return f"{gait}_right"
        if any(f"sideway_{angle}" in name for angle in ("045", "090", "135")):
            return f"{gait}_diagonal"
    if "backward" in name:
        return f"{_gait(name)}_backward"
    if name.startswith("turn_") or "_arc_" in name:
        return f"turn_{_gait(name)}"
    if "forward" in name or "_ff_" in name:
        return f"{_gait(name)}_forward"
    return None


def _base_name(path):
    return re.sub(r"_M(?=\.bvh$)", "", PurePosixPath(path).name)


def _is_mirror(path):
    return PurePosixPath(path).name.endswith("_M.bvh")


def _evenly_spaced(items, count):
    if count >= len(items):
        return list(items)
    if count == 1:
        return [items[0]]
    return [items[round(i * (len(items) - 1) / (count - 1))] for i in range(count)]


def select_balanced_pairs(names, quotas=None):
    """Select only complete original/mirror pairs up to each file quota."""
    quotas = DEFAULT_QUOTAS if quotas is None else quotas
    pairs = defaultdict(lambda: defaultdict(dict))
    for path in sorted(names):
        if not path.lower().endswith(".bvh"):
            continue
        category = classify_motion(path)
        if category not in quotas:
            continue
        pairs[category][_base_name(path)][_is_mirror(path)] = path

    selected = []
    counts = {}
    for category, quota in quotas.items():
        complete = sorted(
            base for base, variants in pairs[category].items()
            if False in variants and True in variants
        )
        chosen = _evenly_spaced(complete, min(len(complete), quota // 2))
        for base in chosen:
            selected.extend([pairs[category][base][False], pairs[category][base][True]])
        counts[category] = 2 * len(chosen)
    return selected, counts


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("archive", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--manifest", type=Path, required=True)
    args = parser.parse_args()

    if args.output.exists() and any(args.output.iterdir()):
        parser.error(f"refusing to write into nonempty output: {args.output}")
    args.output.mkdir(parents=True, exist_ok=True)
    args.manifest.parent.mkdir(parents=True, exist_ok=True)

    with zipfile.ZipFile(args.archive) as archive:
        selected, counts = select_balanced_pairs(archive.namelist())
        for member in selected:
            destination = args.output / PurePosixPath(member).name
            with archive.open(member) as source, destination.open("wb") as target:
                while chunk := source.read(1024 * 1024):
                    target.write(chunk)

    manifest = {
        "archive": str(args.archive),
        "output": str(args.output),
        "selection_policy": "complete original/mirror pairs, evenly sampled by filename",
        "quotas": DEFAULT_QUOTAS,
        "counts": counts,
        "total": len(selected),
        "files": [PurePosixPath(path).name for path in selected],
        "quality_policy": "report only; converted files are never deleted or excluded",
    }
    args.manifest.write_text(json.dumps(manifest, indent=2) + "\n")
    print(json.dumps({"total": len(selected), "counts": counts}, indent=2))


if __name__ == "__main__":
    main()
