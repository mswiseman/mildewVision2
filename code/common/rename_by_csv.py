#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path
import pandas as pd

IMG_EXTS = {".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp", ".webp"}

import re
import unicodedata

import re


def norm_key(x: str) -> str:
    s = str(x).strip()
    # drop extension if present
    s = re.sub(r"\.(png|jpg|jpeg|tif|tiff|bmp|webp)$", "", s, flags=re.IGNORECASE)
    # normalize R1/R2 (and R3 etc if present)
    s = re.sub(r"_R\d+$", "", s, flags=re.IGNORECASE)
    return s

def load_sheet(path: Path) -> pd.DataFrame:
    if path.suffix.lower() in {".xlsx", ".xls"}:
        return pd.read_excel(path)
    elif path.suffix.lower() == ".csv":
        return pd.read_csv(path)
    elif path.suffix.lower() in {".tsv", ".txt"}:
        return pd.read_csv(path, sep="\t")
    else:
        raise ValueError(f"Unsupported spreadsheet type: {path.suffix}")


def build_mapping(df: pd.DataFrame, filename_col: str, pi_col: str) -> dict[str, str]:
    if filename_col not in df.columns:
        raise KeyError(f"Column '{filename_col}' not found in spreadsheet")
    if pi_col not in df.columns:
        raise KeyError(f"Column '{pi_col}' not found in spreadsheet")

    sub = df[[filename_col, pi_col]].copy()
    sub = sub.dropna(subset=[filename_col, pi_col])

    sub[filename_col] = sub[filename_col].astype(str).map(norm_key)
    sub[pi_col] = sub[pi_col].astype(str).str.strip()

    return {norm_key(fn): str(pi).strip() for fn, pi in zip(sub[filename_col], sub[pi_col])}

def unique_path(p: Path) -> Path:
    """Generate unique filename if target exists"""
    if not p.exists():
        return p

    stem = p.stem
    suffix = p.suffix

    i = 1
    while True:
        candidate = p.with_name(f"{stem}_{i}{suffix}")
        if not candidate.exists():
            return candidate
        i += 1


def main():
    parser = argparse.ArgumentParser(
        description="Rename images using filename -> PI_Number mapping from spreadsheet"
    )

    parser.add_argument("--images_dir", required=True, help="Directory with images")
    parser.add_argument("--sheet", required=True, help="Spreadsheet file")
    parser.add_argument("--filename_col", default="filename")
    parser.add_argument("--pi_col", default="PI_Number")

    parser.add_argument("--dry_run", action="store_true")

    parser.add_argument(
        "--on_conflict",
        choices=["skip", "suffix", "overwrite"],
        default="skip",
        help="Behavior when output filename already exists",
    )

    args = parser.parse_args()

    images_dir = Path(args.images_dir).expanduser().resolve()
    sheet_path = Path(args.sheet).expanduser().resolve()

    if not images_dir.exists():
        raise FileNotFoundError(images_dir)

    if not sheet_path.exists():
        raise FileNotFoundError(sheet_path)

    print("Loading spreadsheet...")
    df = load_sheet(sheet_path)

    print("\n--- SHEET INFO ---")
    print("Columns:", list(df.columns))
    print("Row count:", len(df))

    import re

    needle = "005-FryeCanyonFeM_1437_001_R1"

    print("\n--- CONTAINS SEARCH ---")
    pat = re.escape("FryeCanyon")
    m = df["filename"].astype(str).str.contains(pat, na=False)
    print("Rows where filename contains 'FryeCanyon':", int(m.sum()))
    if m.any():
        print(df.loc[m, ["filename", "PI_Number"]].head(20).to_string(index=False))
        print("\nRaw repr of first few matches:")
        for s in df.loc[m, "filename"].astype(str).head(5):
            print(repr(s))

    print("\n--- ALT CONTAINS SEARCH (by number chunk) ---")
    pat2 = re.escape("1437_001")
    m2 = df["filename"].astype(str).str.contains(pat2, na=False)
    print("Rows where filename contains '1437_001':", int(m2.sum()))
    if m2.any():
        print(df.loc[m2, ["filename", "PI_Number"]].head(20).to_string(index=False))
        for s in df.loc[m2, "filename"].astype(str).head(5):
            print(repr(s))


    print("Building filename mapping...")
    mapping = build_mapping(df, args.filename_col, args.pi_col)

    print("Scanning image directory...")
    images = [
        p for p in images_dir.iterdir()
        if p.is_file() and p.suffix.lower() in IMG_EXTS
    ]

    renamed = 0
    skipped_no_match = []
    skipped_conflict = []


    for img in sorted(images):

        base = norm_key(img.stem)
        if base not in mapping:
            print(f"[SKIP no spreadsheet match] {img.name}")
            skipped_no_match.append(img.name)
            continue

        new_name = mapping[base]

        if not new_name:
            print(f"[SKIP missing PI_Number] {img.name}")
            skipped_no_match.append(img.name)
            continue

        target = img.with_name(f"{new_name}{img.suffix.lower()}")

        if target.exists():

            if args.on_conflict == "skip":
                print(f"[SKIP conflict exists] {img.name} -> {target.name}")
                skipped_conflict.append(img.name)
                continue

            elif args.on_conflict == "suffix":
                target = unique_path(target)

            elif args.on_conflict == "overwrite":
                pass

        if args.dry_run:
            print(f"[DRY] {img.name} -> {target.name}")

        else:

            if args.on_conflict == "overwrite" and target.exists():
                target.unlink()

            img.rename(target)

            print(f"[OK] {img.name} -> {target.name}")

        renamed += 1

    print("\n========== SUMMARY ==========")

    print(f"Images found: {len(images)}")
    print(f"Renamed: {renamed}{' (dry-run)' if args.dry_run else ''}")
    print(f"Skipped (no spreadsheet match): {len(skipped_no_match)}")
    print(f"Skipped (conflict): {len(skipped_conflict)}")

    if skipped_no_match:
        print("\nFiles skipped (not found in spreadsheet):")
        for f in skipped_no_match:
            print(f"  {f}")

    if skipped_conflict:
        print("\nFiles skipped (output filename already exists):")
        for f in skipped_conflict:
            print(f"  {f}")


if __name__ == "__main__":
    main()