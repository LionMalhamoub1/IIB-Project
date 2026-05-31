"""
Copy all code and documentation files into a single flat folder, excluding large data/result files.
Each file is renamed to encode its original path (separators replaced with '__').
A _hierarchy.txt file is written listing every file and its original location.
Run from the project root: python export_code.py
"""

import shutil
from pathlib import Path
from datetime import datetime

SRC = Path(__file__).parents[1]

DEST = SRC.parent / f"IIB-Project-code-{datetime.now().strftime('%Y%m%d')}"

SKIP_DIRS = {
    ".git",
    ".claude",
    "__pycache__",
    "data",
    "cache",
    "models",
    "results",
    "results (3month)",
    "outputs",
    "plots",
    "Social Disruptions",
    "Supply Chain Chokepoints",
    "API Costs",
}

KEEP_EXTENSIONS = {
    ".py",
    ".txt",
}


def collect_files(src: Path) -> list[Path]:
    files = []
    for item in sorted(src.rglob("*")):
        if any(part in SKIP_DIRS for part in item.parts[len(src.parts):]):
            continue
        if item.is_dir():
            continue
        if item.suffix.lower() not in KEEP_EXTENSIONS and item.name not in {".gitignore", ".env.example"}:
            continue
        files.append(item)
    return files


def flat_name(rel: Path) -> str:
    """Convert a/b/c.py  →  a__b__c.py"""
    parts = list(rel.parts)
    stem = "__".join(parts[:-1] + [rel.stem])
    return stem + rel.suffix


def write_hierarchy(files: list[Path], src: Path, dest: Path) -> None:
    lines = [
        "FILE HIERARCHY",
        "=" * 60,
        f"Source project : {src}",
        f"Exported on    : {datetime.now().strftime('%Y-%m-%d %H:%M')}",
        "",
        "Each line shows:  <flat filename>  →  <original path>",
        "=" * 60,
        "",
    ]
    for f in files:
        rel = f.relative_to(src)
        lines.append(f"{flat_name(rel):<60}  {rel}")
    (dest / "_hierarchy.txt").write_text("\n".join(lines), encoding="utf-8")


def copy_flat(src: Path, dest: Path) -> None:
    dest.mkdir(parents=True, exist_ok=True)
    files = collect_files(src)

    for f in files:
        rel = f.relative_to(src)
        target = dest / flat_name(rel)
        shutil.copy2(f, target)

    write_hierarchy(files, src, dest)
    print(f"Copied  {len(files)} files  →  {dest}")
    print(f"Hierarchy written to _hierarchy.txt")


if __name__ == "__main__":
    if DEST.exists():
        print(f"Removing existing: {DEST}")
        shutil.rmtree(DEST)

    print(f"Source : {SRC}")
    print(f"Dest   : {DEST}")
    copy_flat(SRC, DEST)
