# !/usr/bin/env python3
from __future__ import annotations

import argparse
import re
import shutil
from pathlib import Path

import numpy as np
import pandas as pd

IMG_EXTS = [".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp", ".webp"]


def choose_top_row(g: pd.DataFrame, value_col: str) -> pd.Series:
    """
    Pick the row with the highest value_col within the group.
    Tie-breaker: first occurrence of the max.
    """
    vals = pd.to_numeric(g[value_col], errors="coerce")
    idx = vals.idxmax()  # ignores NaNs if present; will be NaN if all NaN
    return g.loc[idx]

def parse_dpi(imaging_date: str, dpi_col_value) -> str:
    """
    Return dpi folder name as a string like '5dpi'.
    Prefer the numeric dpi column if present; otherwise parse from imaging_date like '5-25-2025_5dpi'.
    """
    if pd.notna(dpi_col_value):
        try:
            return f"{int(dpi_col_value)}dpi"
        except Exception:
            pass

    if isinstance(imaging_date, str) and "dpi" in imaging_date:
        parts = imaging_date.split("_")
        for p in parts[::-1]:
            if "dpi" in p:
                return p  # e.g. '5dpi'
    return "unknown_dpi"


def choose_median_row(g: pd.DataFrame, value_col: str) -> pd.Series:
    """
    Pick the row whose value_col is closest to the group's median.
    This is robust even when the true median lies between two values.
    """
    vals = pd.to_numeric(g[value_col], errors="coerce")
    med = np.nanmedian(vals.to_numpy())
    dist = (vals - med).abs()
    idx = dist.idxmin()
    return g.loc[idx]


def find_image_file(folder: Path, stem: str) -> Path | None:
    """
    Try to locate an image file within folder for a given filename stem.
    We try:
      - exact stem + any allowed extension
      - any file starting with stem (handles suffixes like '_stacked', etc.)
    """
    if not folder.exists():
        return None

    stem = stem.strip()

    # 1) exact matches
    for ext in IMG_EXTS:
        p = folder / f"{stem}{ext}"
        if p.exists():
            return p

    # 2) startswith matches
    for ext in IMG_EXTS:
        hits = sorted(folder.glob(f"{stem}*{ext}"))
        if hits:
            return hits[0]

    return None


import re
import unicodedata


def safe_windows_name(x: object, default: str) -> str:
    """
    Very strict Windows-safe file name segment:
    - Normalize Unicode (NFKC)
    - Replace any non [A-Za-z0-9._-] with '_'
    - Strip trailing dots/spaces (Windows forbids)
    - Avoid reserved device names
    - If empty/NA-ish -> default
    """
    s = "" if x is None else str(x)
    s = s.strip()

    if s == "" or s.lower() in {"na", "nan", "none", "null", "."}:
        s = default

    # Normalize Unicode to collapse “weird” forms
    s = unicodedata.normalize("NFKC", s)

    # Replace anything outside a conservative safe set
    s = re.sub(r"[^A-Za-z0-9._-]+", "_", s)

    # Strip trailing spaces/dots (illegal on Windows)
    s = s.rstrip(" .")

    reserved = {
        "CON", "PRN", "AUX", "NUL",
        "COM1", "COM2", "COM3", "COM4", "COM5", "COM6", "COM7", "COM8", "COM9",
        "LPT1", "LPT2", "LPT3", "LPT4", "LPT5", "LPT6", "LPT7", "LPT8", "LPT9",
    }
    if s.upper() in reserved:
        s = f"_{s}_"

    if s == "":
        s = re.sub(r"[^A-Za-z0-9._-]+", "_", default).rstrip(" .")
        if s == "":
            s = "unknown"

    return s


def main():
    ap = argparse.ArgumentParser(
        description="Select median-severity image per (pm, imaging_date, modeling_id) and copy to GRIN_Global."
    )
    ap.add_argument("--csv", required=True, help="Path to CSV")
    ap.add_argument(
        "--base",
        default="/g/Stacked/2022-2025_NCGR_Germplasm_Screening",
        help="Base directory containing /[pm]/[imaging_date]/[tray]/images",
    )
    ap.add_argument("--out", default="/g/Stacked/GRIN_Global", help="Output base directory")
    ap.add_argument(
        "--location",
        default="NCGR",
        help="Only process rows where location == this value (set to '' to disable filtering)",
    )
    ap.add_argument("--value_col", default="severity_rate_patch",
                    help="Column used to compute the median representative")
    ap.add_argument("--dry_run", action="store_true", help="Print actions without copying files")
    ap.add_argument(
        "--fallback_id",
        default="modeling_id",
        help="If pi_number is missing/invalid, fall back to this column for naming (e.g., 'usda_number' or 'modeling_id')",
    )
    ap.add_argument(
        "--skip_existing",
        action="store_true",
        help="If destination file already exists, skip instead of overwriting",
    )
    args = ap.parse_args()

    csv_path = Path(args.csv)
    base_dir = Path(args.base)
    out_dir = Path(args.out)

    df = pd.read_csv(csv_path)

    required = ["pm", "imaging_date", "tray", "filename", "modeling_id", args.value_col]
    for c in required:
        if c not in df.columns:
            raise ValueError(f"CSV is missing required column: {c}")

    # Optional columns
    if "pi_number" not in df.columns:
        df["pi_number"] = np.nan
    if "dpi" not in df.columns:
        df["dpi"] = np.nan
    if "location" not in df.columns:
        df["location"] = np.nan

    # Filter to location if requested
    if args.location:
        df = df[df["location"].astype(str).str.strip().eq(args.location)].copy()

    # Ensure numeric values for median selection
    df[args.value_col] = pd.to_numeric(df[args.value_col], errors="coerce")
    df = df.dropna(subset=[args.value_col])

    group_cols = ["pm", "imaging_date", "modeling_id"]
    reps = (
        df.groupby(group_cols, dropna=False, as_index=False)
        .apply(lambda g: choose_top_row(g, args.value_col))
        .reset_index(drop=True)
    )

    print(f"Groups found: {reps.shape[0]:,}")

    copied = 0
    missing_img = 0
    used_fallback = 0
    bad_name = 0
    skipped_existing = 0

    for _, row in reps.iterrows():
        pm = str(row["pm"])
        imaging_date = str(row["imaging_date"])
        tray = str(row["tray"])
        stem = str(row["filename"]).strip()

        dpi_folder = parse_dpi(imaging_date, row.get("dpi", np.nan))

        src_folder = base_dir / pm / imaging_date / tray
        src_img = find_image_file(src_folder, stem)
        if src_img is None:
            missing_img += 1
            print(f"[MISSING] {src_folder}  stem='{stem}'")
            continue

        # Build a robust, Windows-safe filename stem:
        # Prefer pi_number; else fallback_id; else a composite default.
        raw_pi = row.get("pi_number", np.nan)
        raw_fallback = row.get(args.fallback_id, "")

        composite_default = f"{pm}_{imaging_date}_{tray}_{raw_fallback}"
        pi_safe = safe_windows_name(raw_pi, default=safe_windows_name(raw_fallback, default=composite_default))

        # Track if we had to use fallback
        if (raw_pi is None) or (str(raw_pi).strip() == "") or (
                str(raw_pi).strip().lower() in {"na", "nan", "none", "null", "."}):
            used_fallback += 1

        dst_folder = out_dir / "powdery" / pm / dpi_folder
        dst_folder.mkdir(parents=True, exist_ok=True)

        dst_img = dst_folder / f"{pi_safe}{src_img.suffix.lower()}"

        # Hard guard: ensure we have an actual file name
        if dst_img.suffix == "" or str(dst_img.name).strip() == "":
            bad_name += 1
            print(
                f"[SKIP bad-name] pm={pm} imaging_date={imaging_date} tray={tray} "
                f"pi_raw={raw_pi!r} fallback={raw_fallback!r}"
            )
            continue

        if args.skip_existing and dst_img.exists():
            skipped_existing += 1
            print(f"[SKIP exists] {dst_img}")
            continue

        try:
            if args.dry_run:
                print(f"[DRY] copy {src_img} -> {dst_img}")
            else:
                # Use copyfile first (simpler), then copystat (optional)
                shutil.copyfile(str(src_img), str(dst_img))
                # If you want metadata too, uncomment:
                # shutil.copystat(str(src_img), str(dst_img))
            copied += 1
        except OSError as e:
            print(f"[ERROR] {e}")
            print(f"  src_img={src_img}")
            print(f"  dst_img={dst_img}")
            print(f"  dst_name_repr={dst_img.name!r}")
            print(f"  dst_name_codepoints={[ord(c) for c in dst_img.name]}")
            raise

    print("\nDone.")
    print(f"Copied: {copied:,}")
    print(f"Missing images: {missing_img:,}")
    print(f"Used fallback for naming (pi_number missing/invalid): {used_fallback:,}")
    print(f"Skipped due to bad destination name: {bad_name:,}")
    print(f"Skipped existing destination files: {skipped_existing:,}")


if __name__ == "__main__":
    main()