import json
import numpy as np
from pathlib import Path
from pycocotools import mask as mask_utils
import imageio.v2 as imageio


def coco_to_binary_masks(coco_json_path, out_dir, category_ids=None):
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    coco = json.loads(Path(coco_json_path).read_text(encoding="utf-8"))
    anns = coco["annotations"]

    # Map image_id -> (height, width, file_name)
    img_map = {img["id"]: img for img in coco["images"]}

    # Group annotations by image_id
    by_image = {}
    for ann in anns:
        if category_ids is not None and ann.get("category_id") not in set(category_ids):
            continue
        by_image.setdefault(ann["image_id"], []).append(ann)

    for image_id, ann_list in by_image.items():
        img_info = img_map[image_id]
        h, w = img_info["height"], img_info["width"]

        mask = np.zeros((h, w), dtype=np.uint8)

        for ann in ann_list:
            seg = ann.get("segmentation")
            if seg is None:
                continue

            # seg can be polygons (list) or RLE (dict)
            if isinstance(seg, list):
                # polygons -> RLEs -> union
                rles = mask_utils.frPyObjects(seg, h, w)
                rle = mask_utils.merge(rles)
            elif isinstance(seg, dict):
                # already RLE
                rle = seg
            else:
                continue

            m = mask_utils.decode(rle)  # 0/1
            mask[m > 0] = 255

        # name mask after the image file (or image id)
        stem = Path(img_info.get("file_name", str(image_id))).stem
        imageio.imwrite(out_dir / f"{stem}_mask.png", mask)

# Example:
coco_to_binary_masks("annotations.coco.json", "masks_out", category_ids=[1])  # optional filter
