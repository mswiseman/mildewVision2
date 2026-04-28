import os
import cv2
import h5py
import numpy as np

input_dir = r"C:\Users\Intel User\Desktop\blackbird_scripts\data\three_class_test_pngs\test"
output_file = r"C:\Users\Intel User\Desktop\blackbird_scripts\data\testv2.hdf5"
IMG_HEIGHT = 224
IMG_WIDTH = 224

LABEL_MAP = {
    "clear": 0,
    "infected": 1,
    "conidiophore": 2
}


def infer_label_from_filename(filename: str) -> int:
    # Supports either suffix style: "..._clear.png" or your split style "xxx_clear.png"
    base = os.path.splitext(filename)[0].lower()

    # Prefer suffix match
    for k, v in LABEL_MAP.items():
        if base.endswith(f"_{k}"):
            return v

    # Fallback: your original "split on underscore and take [1]" pattern
    parts = base.split("_")
    if len(parts) >= 2 and parts[1] in LABEL_MAP:
        return LABEL_MAP[parts[1]]

    raise ValueError(f"Could not infer label for file: {filename}")


def write_hdf5_streaming(input_dir: str, output_file: str):
    # Collect valid png files first (for pre-allocation)
    files = [fn for fn in os.listdir(input_dir) if fn.lower().endswith(".png")]
    files.sort()

    valid_files = []
    labels = []

    for fn in files:
        try:
            y = infer_label_from_filename(fn)
            valid_files.append(fn)
            labels.append(y)
        except ValueError:
            # skip files that don't match naming
            continue

    if not valid_files:
        raise RuntimeError("No valid labeled .png files found.")

    N = len(valid_files)
    labels = np.asarray(labels, dtype=np.int64)

    # Create HDF5 and pre-allocate datasets, then stream writes
    with h5py.File(output_file, "w") as f:
        # Faster than writing one huge array at the end; chunked for compression + I/O
        img_ds = f.create_dataset(
            "images",
            shape=(N, IMG_HEIGHT, IMG_WIDTH, 3),
            dtype=np.uint8,
            chunks=(64, IMG_HEIGHT, IMG_WIDTH, 3),  # adjust chunk size if needed
            compression="gzip",
            compression_opts=1,  # MUCH faster than 6; try 0-3 for speed
            shuffle=True
        )
        f.create_dataset("labels", data=labels, dtype=np.int64)

        # Optional: keep filenames for debugging/repro
        str_dt = h5py.string_dtype(encoding="utf-8")
        fname_ds = f.create_dataset("filenames", shape=(N,), dtype=str_dt)

        for i, fn in enumerate(valid_files):
            path = os.path.join(input_dir, fn)
            img = cv2.imread(path, cv2.IMREAD_COLOR)
            if img is None:
                raise RuntimeError(f"Failed to read image: {path}")

            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            img = cv2.resize(img, (IMG_WIDTH, IMG_HEIGHT), interpolation=cv2.INTER_AREA)

            img_ds[i] = img
            fname_ds[i] = fn

            # lightweight progress print every 500 images
            if (i + 1) % 500 == 0 or (i + 1) == N:
                print(f"Wrote {i + 1}/{N}")

        print(f"Images dataset shape: {f['images'].shape}")
        print(f"Labels dataset shape: {f['labels'].shape}")
        print(f"Saved to {output_file}")


if __name__ == "__main__":
    write_hdf5_streaming(input_dir, output_file)

