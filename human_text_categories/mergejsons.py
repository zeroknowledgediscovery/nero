#!/usr/bin/env python3
import json
import re
from pathlib import Path
from typing import Any, Dict, Optional

ID_RE_GUTENBERG = re.compile(r"^gutenberg_(\d+)\.json$", re.IGNORECASE)
ID_RE_INDEX     = re.compile(r"^Guten_index(\d+)\.json$", re.IGNORECASE)

def load_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)

def save_json(obj: Any, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)

def extract_id(filename: str, regex: re.Pattern) -> Optional[str]:
    m = regex.match(filename)
    return m.group(1) if m else None

def merge_into_index(index_obj: Any, gutenberg_obj: Any, key: str = "gutenberg") -> Any:
    # Common expected case: index_obj is a dict; store gutenberg_obj under a key
    if isinstance(index_obj, dict):
        # Don’t accidentally overwrite existing key unless you want to
        if key in index_obj:
            # If the key exists, namespace it rather than overwrite
            # (change this behavior if overwrite is preferred)
            index_obj[f"{key}_merged"] = gutenberg_obj
        else:
            index_obj[key] = gutenberg_obj
        return index_obj

    # If index_obj is not a dict, wrap it so we can still augment
    return {"index": index_obj, key: gutenberg_obj}

def main(
    dir_gutenberg: str,
    dir_index: str,
    out_dir: str,
    merge_key: str = "gutenberg",
) -> None:
    d1 = Path(dir_gutenberg)
    d2 = Path(dir_index)
    out = Path(out_dir)

    # Map ID -> path for gutenberg_<ID>.json
    gutenberg_map: Dict[str, Path] = {}
    for p in d1.iterdir():
        if not p.is_file():
            continue
        gid = extract_id(p.name, ID_RE_GUTENBERG)
        if gid is not None:
            gutenberg_map[gid] = p

    updated = 0
    missing = 0
    skipped = 0

    for idx_path in d2.iterdir():
        if not idx_path.is_file():
            continue

        iid = extract_id(idx_path.name, ID_RE_INDEX)
        if iid is None:
            skipped += 1
            continue

        gut_path = gutenberg_map.get(iid)
        if gut_path is None:
            missing += 1
            continue

        index_obj = load_json(idx_path)
        gutenberg_obj = load_json(gut_path)

        merged = merge_into_index(index_obj, gutenberg_obj, key=merge_key)

        out_path = out / idx_path.name
        save_json(merged, out_path)
        updated += 1

    print(f"Updated: {updated}")
    print(f"Missing gutenberg matches: {missing}")
    print(f"Skipped (non-matching filenames): {skipped}")
    print(f"Output written to: {out.resolve()}")

if __name__ == "__main__":
    # EXAMPLE USAGE (edit these paths):
    main(
        dir_gutenberg="Gutenberg_json",
        dir_index="entropy_rates",
        out_dir="augmented_jsons",
        merge_key="gutenberg",
    )
