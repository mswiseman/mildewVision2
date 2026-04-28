#!/usr/bin/env python3
"""
Convert images stored in .hdf5/.h5 files to .png.

Suffix rules:
  0 -> _clear
  1 -> _infected
  2 -> _conidiophore

Tries common dataset names automatically:
  images: ["images", "imgs", "x", "X", "data"]
  labels: ["labels", "y", "Y", "target", "targets", "classes"]

Usage:
  python hdf5_to_png.py --input /path/to/file_or_dir --output /path/to/outdir
  python hdf5_to_png.py --input dataset.h5 --output out --img-key images --label-key labels
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Optional, Tuple, Iterable

import h5py
import numpy as np
from PIL import Image

LABEL_SUFFIX = {
    0: "clear",
    1: "infected",
    2: "conidiophore",
}

DEFAULT_IMAGE_KEYS = ["images", "imgs", "x", "X", "data"]
DEFAULT_LABEL_KEYS = ["labels", "y", "Y", "target", "targets", "classes"]


def find_first_dataset(h5: h5py.File, candidates: list[str]) -> Optional[str]:
    """Return the path of the first dataset whose basename matches one of candidates."""
    found = []

    def visitor(name, obj):
        if isinstance(obj, h5py.Dataset):
            base = name.split("/")[-1]
            if base in candidates:
                found.append(name)

    h5.visititems(visitor)
    return found[0] if found else None


def as_numpy(x) -> np.ndarray:
    arr = np.array(x)
    return arr


def normalize_to_uint8(img: np.ndarray) -> np.ndarray:
    """
    Convert various numeric dtypes/ranges into uint8 for PNG saving.

    Handles:
      - uint8 passthrough
      - float [0,1] or arbitrary range
      - uint16 etc scaled down
    """
    if img.dtype == np.uint8:
        return img

    img = img.astype(np.float32)

    # If it looks like [0,1], scale to [0,255]
    if np.nanmin(img) >= 0.0 and np.nanmax(img) <= 1.0:
        img = img * 255.0
    else:
        # min-max scale per image
        mn = np.nanmin(img)
        mx = np.nanmax(img)
        if mx > mn:
            img = (img - mn) / (mx - mn) * 255.0
        else:
            img = np.zeros_like(img)

    img = np.clip(img, 0, 255).astype(np.uint8)
    return img


def ensure_hwc(img: np.ndarray) -> np.ndarray:
    """
    Make image array HxW or HxWxC for PIL.
    Accepts common layouts:
      - HxW
      - HxWxC
      - CxHxW  (converted)
    """
    if img.ndim == 2:
        return img

    if img.ndim == 3:
        # If first dim is small, assume CHW
        if img.shape[0] in (1, 3, 4) and img.shape[0] < img.shape[-1]:
            img = np.transpose(img, (1, 2, 0))
        return img

    raise ValueError(f"Unsupported image ndim={img.ndim} with shape={img.shape}")


def save_png(img: np.ndarray, out_path: Path) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    img_u8 = normalize_to_uint8(ensure_hwc(img))

    # If single-channel with trailing channel dim (H,W,1), squeeze for nicer PNGs
    if img_u8.ndim == 3 and img_u8.shape[2] == 1:
        img_u8 = img_u8[:, :, 0]

    Image.fromarray(img_u8).save(out_path, format="PNG")


def iter_files(input_path: Path) -> Iterable[Path]:
    if input_path.is_file():
        yield input_path
        return
    for p in sorted(input_path.rglob("*")):
        if p.is_file() and p.suffix.lower() in (".h5", ".hdf5"):
            yield p


def try_pair_datasets(
        h5: h5py.File, img_key: Optional[str], label_key: Optional[str]
) -> Tuple[h5py.Dataset, h5py.Dataset]:
    """
    Find datasets for images and labels.
    If keys are provided, use those (as paths or basenames).
    Otherwise, search common names.
    """
    if img_key is None:
        img_path = find_first_dataset(h5, DEFAULT_IMAGE_KEYS)
    else:
        img_path = img_key if img_key in h5 else find_first_dataset(h5, [img_key])

    if label_key is None:
        lab_path = find_first_dataset(h5, DEFAULT_LABEL_KEYS)
    else:
        lab_path = label_key if label_key in h5 else find_first_dataset(h5, [label_key])

    if not img_path or not lab_path:
        raise KeyError(
            "Could not auto-detect image/label datasets.\n"
            f"Found image dataset path: {img_path}\n"
            f"Found label dataset path: {lab_path}\n\n"
            "Try specifying --img-key and --label-key.\n"
            "Examples:\n"
            "  --img-key images --label-key labels\n"
            "  --img-key /group/images --label-key /group/labels"
        )

    img_ds = h5[img_path]
    lab_ds = h5[lab_path]
    return img_ds, lab_ds


def extract_and_save_from_paired_datasets(
        img_ds: h5py.Dataset,
        lab_ds: h5py.Dataset,
        out_dir: Path,
        base_stem: str,
        limit: Optional[int] = None,
) -> int:
    imgs = img_ds
    labs = lab_ds

    if imgs.shape[0] != labs.shape[0]:
        raise ValueError(f"Images and labels length mismatch: {imgs.shape[0]} vs {labs.shape[0]}")

    n = imgs.shape[0]
    if limit is not None:
        n = min(n, limit)

    count = 0
    for i in range(n):
        img = as_numpy(imgs[i])
        label_raw = as_numpy(labs[i]).item() if np.array(labs[i]).size == 1 else int(as_numpy(labs[i])[0])
        suffix = LABEL_SUFFIX.get(int(label_raw), f"label{int(label_raw)}")

        out_path = out_dir / f"{base_stem}_{i:06d}_{suffix}.png"
        save_png(img, out_path)
        count += 1

    return count


def extract_and_save_from_groups(
        h5: h5py.File,
        out_dir: Path,
        base_stem: str,
        limit: Optional[int] = None,
) -> int:
    """
    Fallback: iterate groups and look for datasets named 'image'/'label' or similar.
    """
    image_names = {"image", "img", "x", "data"}
    label_names = {"label", "y", "class", "target"}

    items = []

    def visitor(name, obj):
        if isinstance(obj, h5py.Group):
            # group-level scan for image+label datasets
            keys = set(obj.keys())
            img_k = next((k for k in obj.keys() if k in image_names), None)
            lab_k = next((k for k in obj.keys() if k in label_names), None)
            if img_k and lab_k:
                items.append((name, img_k, lab_k))

    h5.visititems(visitor)

    if not items:
        raise KeyError(
            "Fallback group scan failed: couldn't find groups containing both an image dataset "
            "(e.g. 'image'/'img'/'x'/'data') and a label dataset (e.g. 'label'/'y'/'class'/'target')."
        )

    if limit is not None:
        items = items[:limit]

    count = 0
    for idx, (grp_path, img_k, lab_k) in enumerate(items):
        grp = h5[grp_path]
        img = as_numpy(grp[img_k][()])
        label_raw = as_numpy(grp[lab_k][()]).item() if np.array(grp[lab_k][()]).size == 1 else int(
            as_numpy(grp[lab_k][()])[0])
        suffix = LABEL_SUFFIX.get(int(label_raw), f"label{int(label_raw)}")

        out_path = out_dir / f"{base_stem}_{idx:06d}_{suffix}.png"
        save_png(img, out_path)
        count += 1

    return count


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", required=True, help="Path to a .h5/.hdf5 file or a directory containing them")
    ap.add_argument("--output", required=True, help="Output directory for PNGs")
    ap.add_argument("--img-key", default=None, help="Dataset path or basename for images (optional)")
    ap.add_argument("--label-key", default=None, help="Dataset path or basename for labels (optional)")
    ap.add_argument("--limit", type=int, default=None, help="Optional cap on number of images per file")
    args = ap.parse_args()

    in_path = Path(args.input).expanduser().resolve()
    out_dir = Path(args.output).expanduser().resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    total = 0
    for f in iter_files(in_path):
        base_stem = f.stem
        file_out = out_dir / base_stem
        file_out.mkdir(parents=True, exist_ok=True)

        with h5py.File(f, "r") as h5:
            # First attempt: paired datasets
            try:
                img_ds, lab_ds = try_pair_datasets(h5, args.img_key, args.label_key)
                n = extract_and_save_from_paired_datasets(img_ds, lab_ds, file_out, base_stem, args.limit)
            except Exception:
                # Fallback: group scan
                n = extract_and_save_from_groups(h5, file_out, base_stem, args.limit)

        print(f"{f.name}: wrote {n} PNGs to {file_out}")
        total += n

    print(f"Done. Total PNGs written: {total}")


if __name__ == "__main__":
    main()

# python hdf5_to_file.py --input "C:\Users\Intel User\Desktop\blackbird_scripts\data\2-2-2024_3_class_mapp_pop_test.hdf5" --output "C:\Users\Intel User\Desktop\blackbird_scripts\data\three_class_test_pngs"
