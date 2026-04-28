import os
import subprocess
import argparse
import concurrent.futures


def parse_args():
    parser = argparse.ArgumentParser(description="Batch convert TIFF/TIF images to PNG using NConvert.")
    parser.add_argument('--input_dir', type=str, required=True,
                        help='Directory containing TIFF/TIF images (recursively searched)')
    parser.add_argument('--output_dir', type=str, required=True,
                        help='Directory to write PNGs')
    parser.add_argument('--nconvert_path', type=str, default="C:/XnView/nconvert.exe",
                        help='Path to NConvert executable')
    parser.add_argument('--max_workers', type=int, default=4,
                        help='Number of parallel processes to run')
    parser.add_argument('--flat', action='store_true',
                        help='Do not preserve folder structure; write all PNGs into output_dir')
    parser.add_argument('--delete_source', action='store_true',
                        help='Delete source TIFF after successful conversion')
    parser.add_argument('--overwrite', action='store_true',
                        help='Overwrite existing PNG outputs')
    return parser.parse_args()


def iter_tiffs(input_dir: str):
    for root, _, files in os.walk(input_dir):
        for fn in files:
            if fn.lower().endswith(('.tif', '.tiff')):
                yield os.path.join(root, fn)


def build_output_path(input_path: str, input_dir: str, output_dir: str, flat: bool):
    base = os.path.splitext(os.path.basename(input_path))[0] + ".png"
    if flat:
        return os.path.join(output_dir, base)

    rel_dir = os.path.relpath(os.path.dirname(input_path), input_dir)
    return os.path.join(output_dir, rel_dir, base)


def convert_one(tiff_path: str, input_dir: str, output_dir: str, nconvert_path: str,
                flat: bool, delete_source: bool, overwrite: bool):
    png_path = build_output_path(tiff_path, input_dir, output_dir, flat)
    os.makedirs(os.path.dirname(png_path), exist_ok=True)

    if (not overwrite) and os.path.exists(png_path):
        return f"SKIP (exists): {png_path}"

    # NConvert: input file is last argument; -o specifies output file
    subprocess.run(
        [nconvert_path, "-out", "png", "-o", png_path, tiff_path],
        check=True,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL
    )

    if delete_source:
        os.remove(tiff_path)

    return f"OK: {tiff_path} -> {png_path}"


def main():
    args = parse_args()

    if not os.path.isfile(args.nconvert_path):
        raise FileNotFoundError(f"NConvert not found: {args.nconvert_path}")

    os.makedirs(args.output_dir, exist_ok=True)

    tiffs = list(iter_tiffs(args.input_dir))
    if not tiffs:
        print(f"No .tif/.tiff files found under: {args.input_dir}")
        return

    print(f"Found {len(tiffs)} TIFF(s). Converting with {args.max_workers} workers...")

    # Use ProcessPool for parallel external subprocess calls
    with concurrent.futures.ProcessPoolExecutor(max_workers=args.max_workers) as ex:
        futures = [
            ex.submit(
                convert_one,
                t, args.input_dir, args.output_dir, args.nconvert_path,
                args.flat, args.delete_source, args.overwrite
            )
            for t in tiffs
        ]

        done = 0
        for fut in concurrent.futures.as_completed(futures):
            done += 1
            try:
                msg = fut.result()
                # Print occasionally to avoid huge console spam if you have many files
                if done <= 20 or done % 50 == 0 or msg.startswith("SKIP"):
                    print(msg)
            except subprocess.CalledProcessError as e:
                print(f"ERROR converting a file: {e}")

    print("=== Conversion complete ===")


if __name__ == "__main__":
    main()

# python convert_tif_to_png.py --input_dir "F:/Stacked/June_10_2022_V6/6-20-2022_10dpi/1" --output_dir "F:/Stacked/June_10_2022_V6/6-15-2022_5dpi/1" --max_workers 1