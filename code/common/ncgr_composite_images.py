#!/usr/bin/env python3
from __future__ import annotations

import argparse
import math
import os
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple
from collections import defaultdict

from PIL import Image, ImageDraw, ImageFont

IMG_EXTS = {".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp", ".webp"}


def fix_path(p: str | Path) -> Path:
    """
    Convert Git-Bash/MSYS paths like /h/GRIN_Global/... or /c/Users/...
    into Windows drive paths H:/GRIN_Global/... or C:/Users/...
    Prevents accidentally using H:/h/... (a literal 'h' folder).
    """
    s = str(p)

    # Already a Windows drive path
    if re.match(r"^[A-Za-z]:[\\/]", s):
        return Path(s)

    # MSYS-style /h/... or /c/...
    m = re.match(r"^/([a-zA-Z])/(.*)$", s)
    if m:
        drive = m.group(1).upper()
        rest = m.group(2)
        # Use forward slashes; Path handles it fine on Windows
        return Path(f"{drive}:/{rest}")

    # Fallback
    return Path(s)


@dataclass(frozen=True)
class Source:
    key: str
    label: str
    folder: Path


def list_images(folder: Path) -> List[Path]:
    if not folder.is_dir():
        return []
    return [p for p in folder.iterdir() if p.is_file() and p.suffix.lower() in IMG_EXTS]


def choose_best(paths: List[Path]) -> Path:
    pref = {".png": 0, ".tif": 1, ".tiff": 1, ".jpg": 2, ".jpeg": 2, ".webp": 3, ".bmp": 4}

    def score(p: Path) -> Tuple[int, int]:
        return (pref.get(p.suffix.lower(), 9), -p.stat().st_size)

    return sorted(paths, key=score)[0]


def build_index(sources: List[Source]) -> Dict[str, Dict[str, Path]]:
    acc_to_paths: Dict[str, Dict[str, List[Path]]] = {}

    for src in sources:
        imgs = list_images(src.folder)
        for p in imgs:
            acc = p.stem
            acc_to_paths.setdefault(acc, {}).setdefault(src.key, []).append(p)

    resolved: Dict[str, Dict[str, Path]] = {}
    for acc, per_src in acc_to_paths.items():
        resolved[acc] = {}
        for skey, plist in per_src.items():
            resolved[acc][skey] = choose_best(plist)

    return resolved


def load_font(font_size: int) -> ImageFont.ImageFont:
    for name in ["arial.ttf", "Arial.ttf", "DejaVuSans.ttf"]:
        try:
            return ImageFont.truetype(name, font_size)
        except Exception:
            pass
    return ImageFont.load_default()


def fit_image(im: Image.Image, w: int, h: int) -> Image.Image:
    im = im.convert("RGB")
    iw, ih = im.size
    scale = min(w / iw, h / ih)
    nw, nh = max(1, int(iw * scale)), max(1, int(ih * scale))
    resized = im.resize((nw, nh), Image.Resampling.LANCZOS)
    canvas = Image.new("RGB", (w, h), (255, 255, 255))
    ox = (w - nw) // 2
    oy = (h - nh) // 2
    canvas.paste(resized, (ox, oy))
    return canvas


def resize_keep_aspect(im: Image.Image, target_w: int) -> Image.Image:
    """Resize to target_w, keeping aspect ratio (no padding, no cropping)."""
    im = im.convert("RGB")
    iw, ih = im.size
    scale = target_w / float(iw)
    target_h = max(1, int(round(ih * scale)))
    return im.resize((target_w, target_h), Image.Resampling.LANCZOS)


def draw_label(tile: Image.Image, text: str, font: ImageFont.ImageFont, pad: int = 8) -> Image.Image:
    tile = tile.copy()
    draw = ImageDraw.Draw(tile)

    try:
        bbox = draw.textbbox((0, 0), text, font=font)
        tw, th = bbox[2] - bbox[0], bbox[3] - bbox[1]
    except Exception:
        tw, th = draw.textsize(text, font=font)

    band_h = th + pad * 2
    w, h = tile.size
    y0 = h - band_h

    draw.rectangle([0, y0, w, h], fill=(0, 0, 0))
    x = max(pad, (w - tw) // 2)
    y = y0 + pad
    draw.text((x, y), text, fill=(255, 255, 255), font=font)
    return tile


def draw_label_overlay(im: Image.Image, text: str, font: ImageFont.ImageFont, pad: int = 14):
    """
    Overlay label at the TOP of the image so panels touch without whitespace.
    """
    im = im.copy()

    draw = ImageDraw.Draw(im)

    try:
        bbox = draw.textbbox((0, 0), text, font=font)
        tw = bbox[2] - bbox[0]
        th = bbox[3] - bbox[1]
    except:
        tw, th = draw.textsize(text, font=font)

    band_h = th + pad * 2
    w, h = im.size

    # Create semi-transparent band
    overlay = Image.new("RGBA", (w, band_h), (0, 0, 0, 160))
    base = im.convert("RGBA")

    base.paste(overlay, (0, 0), overlay)

    draw2 = ImageDraw.Draw(base)

    x = max(pad, (w - tw) // 2)
    y = pad

    draw2.text((x, y), text, fill=(255, 255, 255, 255), font=font)

    return base.convert("RGB")

def make_composite(
        src_order: List[Source],
        paths_for_acc: Dict[str, Path],
        tile_w: int,
        cols: int,
        margin: int,
        label_font_size: int,
        include_missing: bool,
) -> Image.Image:
    font = load_font(label_font_size)

    # Build tiles (all same width; height determined by aspect ratio)
    tiles: List[Image.Image] = []
    for src in src_order:
        p = paths_for_acc.get(src.key)
        if p is None:
            if not include_missing:
                continue
            # If you include missing, make a placeholder tile at the expected height
            # using the reference aspect ratio 5502/8254
            expected_h = int(round(tile_w * (5502 / 8254)))
            blank = Image.new("RGB", (tile_w, expected_h), (245, 245, 245))
            blank = draw_label_overlay(blank, f"{src.label} (missing)", font)
            tiles.append(blank)
        else:
            im = Image.open(p)
            tile = resize_keep_aspect(im, tile_w)
            tile = draw_label_overlay(tile, src.label, font)
            tiles.append(tile)

    n = len(tiles)
    if n == 0:
        return Image.new("RGB", (tile_w, tile_w), (255, 255, 255))

    cols = min(cols, n)
    rows = math.ceil(n / cols)

    # All tiles should have the same height if originals share aspect ratio
    tile_h = tiles[0].size[1]

    out_w = margin * (cols + 1) + tile_w * cols
    out_h = margin * (rows + 1) + tile_h * rows
    out = Image.new("RGB", (out_w, out_h), (255, 255, 255))

    for i, tile in enumerate(tiles):
        r = i // cols
        c = i % cols
        x = margin + c * (tile_w + margin)
        y = margin + r * (tile_h + margin)
        out.paste(tile, (x, y))

    return out


def main() -> None:
    ap = argparse.ArgumentParser(description="Create labeled composites of matching accessions across folders.")
    ap.add_argument("--out_dir", default="H:/GRIN_Global/composites", help="Output directory for composites")
    ap.add_argument("--min_sources", type=int, default=2, help="Minimum #folders required to output a composite")
    ap.add_argument("--tile_w", type=int, default=700)
    ap.add_argument("--tile_h", type=int, default=700)
    ap.add_argument("--cols", type=int, default=4)
    ap.add_argument("--margin", type=int, default=20)
    ap.add_argument("--label_font_size", type=int, default=28)
    ap.add_argument("--include_missing", action="store_true")
    ap.add_argument("--limit", type=int, default=0)
    args = ap.parse_args()

    sources = [
        Source("abax0", "Abaxial surface (0 dpi)", fix_path("/h/GRIN_Global/abaxial_surface_0dpi")),
        Source("adax0", "Adaxial surface (0 dpi)", fix_path("/h/GRIN_Global/adaxial_surface_0dpi")),
        Source("downy6", "Downy (6 dpi)", fix_path("/h/GRIN_Global/downy/6dpi")),
        Source("pm663_10", "Powdery HPM-663 (10 dpi)", fix_path("/h/GRIN_Global/powdery/HPM-663/10dpi")),
        Source("pm666_10", "Powdery HPM-666 (10 dpi)", fix_path("/h/GRIN_Global/powdery/HPM-666/10dpi")),
        Source("pm1269_10", "Powdery HPM-1269 (10 dpi)", fix_path("/h/GRIN_Global/powdery/HPM-1269/10dpi")),
        Source("pm204_5", "Powdery HPM-204 (5 dpi)", fix_path("/h/GRIN_Global/powdery/HPM-204/5dpi")),
        Source("pm1285_5", "Powdery HPM-1285 (5 dpi)", fix_path("/h/GRIN_Global/powdery/HPM-1285/5dpi")),
    ]

    print("\n--- PATH CHECK ---")
    for s in sources:
        print(f"{s.key:10s} exists={s.folder.exists():5}  path={s.folder}")

    out_dir = fix_path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    index = build_index(sources)

    # Count how many sources per accession
    eligible = [(acc, per_src) for acc, per_src in index.items() if len(per_src) >= args.min_sources]
    eligible.sort(key=lambda x: x[0])

    print(f"\nAccessions total: {len(index)}")
    print(f"Eligible (>= {args.min_sources} folders): {len(eligible)}")

    report_path = out_dir / "composites_index.csv"
    with report_path.open("w", encoding="utf-8") as f:
        f.write("accession,num_sources," + ",".join([s.key for s in sources]) + "\n")

        written = 0
        for acc, per_src in eligible:
            if args.limit and written >= args.limit:
                break

            comp = make_composite(
                src_order=sources,
                paths_for_acc=per_src,
                tile_w=args.tile_w,
                cols=args.cols,
                margin=args.margin,
                label_font_size=args.label_font_size,
                include_missing=args.include_missing,
            )

            out_path = out_dir / f"{acc}_composite.jpg"
            comp.save(out_path, quality=92)

            row = [acc, str(len(per_src))]
            for s in sources:
                row.append(str(per_src.get(s.key, "")))
            f.write(",".join([r.replace(",", ";") for r in row]) + "\n")

            print(f"[OK] {acc}: {len(per_src)} source(s) -> {out_path}")
            written += 1

    print(f"\nDone. Wrote {written} composites.")
    print(f"Index CSV: {report_path}")


if __name__ == "__main__":
    main()