#!/usr/bin/env python3

import argparse
import csv
import random
from pathlib import Path

import cv2
import numpy as np
from PIL import Image

IMG_EXTS = {".png", ".jpg", ".jpeg", ".tif", ".tiff"}


def leaf_mask(img, rel_th):
    """
        Fill holes in a binary image using cv2
        # Relative thresholding (0 < rth < 1)
        #
    Args:
        img:        A RGB format image with PIL format
        rel_th:     Relative threshold value
    Return:
        im_out:     Foreground mask with Numpy array format
    """
    img = np.asarray(img)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    wsize = 50
    wstep = 6

    res_offset = int(wsize / wstep)
    # center_offset = wsize / 2

    height, width, _ = img.shape
    aux_height = int((height / wstep) - res_offset)
    aux_width = int((width / wstep) - res_offset)

    # Output focal metric image
    fmat = np.ndarray(shape=(aux_height, aux_width), dtype=np.float64)

    coor_x = coor_y = 0

    # Get sharpness/focused of the image by calculating the standard deviation
    # Using sliding window
    for row in range(aux_height):
        for col in range(aux_width):
            kernel = img[coor_y: coor_y + wsize, coor_x: coor_x + wsize, :]
            _, stddev = cv2.meanStdDev(kernel)
            maxfm = stddev[0]
            for i in range(1, 3):
                maxfm = max(maxfm, stddev[i])
            fmat[row, col] = maxfm
            coor_x = coor_x + wstep
        coor_x = 0
        coor_y = coor_y + wstep

    _, max_val, _, _ = cv2.minMaxLoc(fmat)
    # 1st Threshold
    _, imbin = cv2.threshold(fmat, max_val * rel_th,
                             255, cv2.THRESH_BINARY_INV)
    imbin = imbin.astype('uint8')
    # Calculate the mean between mask and thresholded mask
    fv = cv2.mean(fmat, imbin)
    th = fv[0]
    # 2nd Threshold
    _, imbin = cv2.threshold(fmat, th, 255, cv2.THRESH_BINARY_INV)
    imbin = imbin.astype('uint8')
    # Bitwise not
    imbin1 = cv2.bitwise_not(imbin)
    imbin1 = imfill(imbin1)
    # Erode
    imbin1 = cv2.erode(imbin1, kernel=np.ones(
        (3, 3), dtype=np.uint8), anchor=(-1, -1), iterations=10)

    h, w = imbin1.shape
    # Add margins for erosion
    for col in range(w):
        imbin1[0, col] = 0
        imbin1[h - 1, col] = 0
    for row in range(h):
        imbin1[row, 0] = 0
        imbin1[row, w - 1] = 0
    imbin1 = cv2.erode(imbin1, kernel=np.ones(
        (3, 3), dtype=np.uint8), anchor=(-1, -1), iterations=4)
    imbin1 = imfill(imbin1)

    # Mask validation
    # The stats var is [num_of_connected_component, 5 (left, top, width, height, area)]
    retval, labels, stats, _ = cv2.connectedComponentsWithStats(
        imbin1, connectivity=8, ltype=cv2.CV_16U)

    # No leaf pixels found
    if retval < 2:
        print('No leaf pixels found. No mask.')
        return None

    # Leaf area so small, skip
    big_l = 1
    for i in range(1, retval):
        if stats[i][4] > stats[big_l][4]:
            big_l = i
    if stats[big_l][4] < (h * w * 0.15):
        print('Leaf area so small! No mask.')
        return None

    coor_x = coor_y = 0
    # Delete other objects, keep the biggest
    for row in range(aux_height):
        for col in range(aux_width):
            imbin1[row, col] = 255 if labels[row, col] == big_l else 0

    imbin1 = cv2.erode(imbin1, kernel=np.ones(
        (3, 3), dtype=np.uint8), anchor=(-1, -1), iterations=10)
    imbin1 = cv2.resize(imbin1, dsize=(width, height), fx=0, fy=0,
                        interpolation=cv2.INTER_NEAREST)

    return imbin1


def imfill(inmask):
    """
        Do floodFill operation on the input mask
    """
    flood_mask = inmask.copy()
    h, w = flood_mask.shape
    mask = np.zeros((h + 2, w + 2), np.uint8)

    for col in range(w):
        if flood_mask[0, col] == 0:
            cv2.floodFill(flood_mask, mask, (col, 0),
                          newVal=255, loDiff=10, upDiff=10)
        if flood_mask[h - 1, col] == 0:
            cv2.floodFill(flood_mask, mask, (col, h - 1),
                          newVal=255, loDiff=10, upDiff=10)

    for row in range(h):
        if flood_mask[row, 0] == 0:
            cv2.floodFill(flood_mask, mask, (0, row),
                          newVal=255, loDiff=10, upDiff=10)
        if flood_mask[row, w - 1] == 0:
            cv2.floodFill(flood_mask, mask, (w - 1, row),
                          newVal=255, loDiff=10, upDiff=10)

    out_mask = inmask.copy()

    for row in range(h):
        for col in range(w):
            if flood_mask[row, col] == 0:
                out_mask[row, col] = 255

    return out_mask


def on_focus(imask, th=0.7):
    """
        Check image patches focused or not
    Args:
        imask:  Image patches' mask
    """
    mask_ratio = np.mean(imask)
    if mask_ratio > th:
        focused = True
    else:
        focused = False

    return focused

def iter_images(images_dir: Path):
    for p in sorted(images_dir.iterdir()):
        if p.suffix.lower() in IMG_EXTS:
            yield p


def grid_positions(w, h, patch):
    xs = range(0, w - patch + 1, patch)
    ys = range(0, h - patch + 1, patch)
    return [(x, y) for y in ys for x in xs]


def sample_patches(
        img_arr,
        mask_arr,
        patch_size,
        n_patches,
        min_leaf_fraction,
        rng
):
    h, w = img_arr.shape[:2]
    coords = []

    for (x, y) in grid_positions(w, h, patch_size):
        patch_mask = mask_arr[y:y + patch_size, x:x + patch_size]
        if patch_mask.size == 0:
            continue
        leaf_fraction = np.mean(patch_mask > 0)
        if leaf_fraction >= min_leaf_fraction:
            coords.append((x, y))

    if len(coords) == 0:
        return []

    if len(coords) <= n_patches:
        rng.shuffle(coords)
        return coords

    return rng.sample(coords, n_patches)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--images_dir", required=True)
    ap.add_argument("--out_dir", required=True)
    ap.add_argument("--patch_size", type=int, default=224)
    ap.add_argument("--n_patches", type=int, default=100)
    ap.add_argument("--rel_th", type=float, default=0.35,
                    help="Relative threshold for leaf_mask()")
    ap.add_argument("--min_leaf_fraction", type=float, default=0.80,
                    help="Minimum fraction of leaf pixels per patch")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    images_dir = Path(args.images_dir)
    out_dir = Path(args.out_dir)
    patches_dir = out_dir / "patches"
    patches_dir.mkdir(parents=True, exist_ok=True)

    rng = random.Random(args.seed)

    manifest_path = out_dir / "patch_manifest.csv"
    with open(manifest_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["source_image", "patch_file", "x", "y"])

        for img_path in iter_images(images_dir):
            img_pil = Image.open(img_path).convert("RGB")
            img_arr = np.asarray(img_pil)

            mask = leaf_mask(img_pil, rel_th=args.rel_th)
            if mask is None:
                print(f"{img_path.name}: no valid leaf mask, skipping")
                continue

            coords = sample_patches(
                img_arr=img_arr,
                mask_arr=mask,
                patch_size=args.patch_size,
                n_patches=args.n_patches,
                min_leaf_fraction=args.min_leaf_fraction,
                rng=rng
            )

            for i, (x, y) in enumerate(coords):
                patch = img_arr[y:y + args.patch_size, x:x + args.patch_size]
                patch_name = f"{img_path.stem}__x{x}_y{y}__{i:03d}.png"
                patch_path = patches_dir / patch_name
                cv2.imwrite(str(patch_path), cv2.cvtColor(patch, cv2.COLOR_RGB2BGR))
                writer.writerow([img_path.name, patch_name, x, y])

            print(f"{img_path.name}: saved {len(coords)} patches")

    print(f"Done. Manifest saved to {manifest_path}")


if __name__ == "__main__":
    main()


#   python random_100_patch_sample.py --images_dir /c/Users/Intel\ User/Downloads/test_set --out_dir /c/Users/Intel\ User/Downloads/test_set/random_patches --n_patches 100 --rel_th 0.35