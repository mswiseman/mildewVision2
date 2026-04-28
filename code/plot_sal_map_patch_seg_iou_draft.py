import os
import time
import argparse
import warnings
from datetime import datetime
from pathlib import Path
from collections import defaultdict

import numpy as np
import pandas as pd
from PIL import Image
from matplotlib import pyplot as plt
from matplotlib.colors import LinearSegmentedColormap

import torch
import torch.nn.functional as F
from torchvision import transforms as tvtrans

from analyzer_config import IMG_HEIGHT, IMG_WIDTH  # kept for consistency

from classification.utils import (
    timeSince, printArgs, load_model, parse_model, set_logging, adaptive_threshold
)
from visualization.viz_helper import (
    get_first_conv_layer, get_last_conv_layer, normalize_image_attr, make_single_hue_cmap
)
from sanity_check.utils import get_saliency_methods, get_saliency_masks

# optional (only needed if you want pixel SR like leaf script)
from metric import pixel_sr1, patch_sr

np.random.seed(2020)
warnings.filterwarnings("ignore", category=UserWarning, module="captum.attr._core.deep_lift")

# -----------------------------
# IoU helpers
# -----------------------------
MASK_EXTS = (".png", ".jpg", ".jpeg", ".tif", ".tiff", ".gif", ".bmp")

CMAP_HYPHAE = make_single_hue_cmap("hyphae_teal", (0.0, 1.0, 1.0))  # teal
CMAP_SPOR = make_single_hue_cmap("spor_magenta", (1.0, 0.0, 1.0))  # magenta

def find_mask_for_image(img_name: str, mask_dir: Path) -> Path | None:
    """
    Match masks by stem:
      - <stem>.<ext>
      - <stem>_mask.<ext>
      - <stem>-mask.<ext>
    """
    if mask_dir is None:
        return None
    mask_dir = Path(mask_dir)
    stem = Path(img_name).stem

    candidates = []
    for ext in MASK_EXTS:
        candidates.append(mask_dir / f"{stem}{ext}")
        candidates.append(mask_dir / f"{stem}_mask{ext}")
        candidates.append(mask_dir / f"{stem}-mask{ext}")

    for p in candidates:
        if p.exists():
            return p
    return None


def sweep_thresholds(heat: np.ndarray, gt_mask: np.ndarray, mode: str,
                     percentiles: list[float], fixed: list[float], pick_metric='iou'):
    """
    heat: float HxW (assumed already normalized 0..1 by normalize_image_attr)
    gt_mask: bool HxW
    mode:
      - 'percentile': threshold by keeping top (100-p)% pixels
      - 'fixed': threshold by heat >= t
    Returns:
      best_row (dict) and all_rows (list[dict])
    """
    rows = []

    if mode == "percentile":
        # interpret p as "percentile cutoff": keep heat >= q where q = percentile(heat, p)
        for p in percentiles:
            thr = float(np.percentile(heat, p))
            pred = (heat >= thr).astype(np.uint8)
            m = bin_metrics(pred, gt_mask)
            rows.append({"sweep_mode": "percentile", "p": float(p), "thr": thr, **m})
    else:
        for thr in fixed:
            thr = float(thr)
            pred = (heat >= thr).astype(np.uint8)
            m = bin_metrics(pred, gt_mask)
            rows.append({"sweep_mode": "fixed", "p": np.nan, "thr": thr, **m})

    key = pick_metric
    best = max(rows, key=lambda r: r[key])
    return best, rows


def load_binary_mask(mask_path: Path, size=(224, 224)) -> np.ndarray:
    """
    Load mask as boolean array HxW.
    - converts to grayscale
    - resizes with NEAREST to preserve labels
    - binarizes as >0
    """
    m = Image.open(mask_path).convert("L")
    if m.size != size:
        m = m.resize(size, resample=Image.NEAREST)
    arr = np.asarray(m)
    return (arr > 0)


def bin_metrics(pred_bin: np.ndarray, gt_bin: np.ndarray) -> dict:
    """
    pred_bin, gt_bin: boolean arrays HxW (or uint8 {0,1}).
    Returns IoU, Dice, Precision, Recall, and areas.
    """
    pred = pred_bin.astype(bool)
    gt = gt_bin.astype(bool)

    inter = np.logical_and(pred, gt).sum()
    union = np.logical_or(pred, gt).sum()
    pred_area = pred.sum()
    gt_area = gt.sum()

    # edge cases: both empty -> perfect
    if union == 0:
        iou = 1.0
    else:
        iou = float(inter) / float(union)

    denom_dice = pred_area + gt_area
    dice = 1.0 if denom_dice == 0 else (2.0 * float(inter) / float(denom_dice))

    precision = 1.0 if pred_area == 0 else (float(inter) / float(pred_area))
    recall = 1.0 if gt_area == 0 else (float(inter) / float(gt_area))

    return {
        "iou": iou,
        "dice": dice,
        "precision": precision,
        "recall": recall,
        "pred_area": int(pred_area),
        "gt_area": int(gt_area),
        "inter": int(inter),
        "union": int(union),
    }


def iter_patch_images_from_folder(root: Path, exts=(".png", ".jpg", ".jpeg", ".tif", ".tiff")):
    root = Path(root)
    for p in sorted(root.rglob("*")):
        if p.is_file() and p.suffix.lower() in exts:
            yield p.name, p


def iter_patch_images_from_hdf5(h5_path: Path):
    """
    Minimal, defensive HDF5 patch loader.
    Expected common patterns:
      - images: (N,H,W,C) or (N,C,H,W) uint8/float
      - filenames: (N,) bytes/str (optional)
    """
    import h5py  # local import so folder mode doesn’t require it

    h5_path = Path(h5_path)
    with h5py.File(h5_path, "r") as f:
        image_keys = ["images", "imgs", "X", "data"]
        name_keys = ["filenames", "names", "ids", "image_names"]

        img_ds = None
        for k in image_keys:
            if k in f:
                img_ds = f[k]
                break
        if img_ds is None:
            for k in f.keys():
                if hasattr(f[k], "shape") and len(f[k].shape) >= 3:
                    img_ds = f[k]
                    break
        if img_ds is None:
            raise ValueError(f"No image dataset found in {h5_path} (tried {image_keys}).")

        name_ds = None
        for k in name_keys:
            if k in f:
                name_ds = f[k]
                break

        n = img_ds.shape[0]
        for i in range(n):
            arr = img_ds[i]
            if name_ds is not None:
                nm = name_ds[i]
                if isinstance(nm, (bytes, np.bytes_)):
                    nm = nm.decode("utf-8", errors="ignore")
                else:
                    nm = str(nm)
            else:
                nm = f"{h5_path.stem}_idx{i:06d}.png"

            yield nm, arr


def to_uint8_hwc(arr: np.ndarray) -> np.ndarray:
    """Convert input to HWC uint8 for visualization + PIL."""
    a = np.array(arr)
    if a.ndim == 2:
        a = np.stack([a, a, a], axis=-1)
    if a.ndim == 3 and a.shape[0] in (1, 3) and a.shape[-1] not in (1, 3):
        a = np.transpose(a, (1, 2, 0))
    if a.dtype != np.uint8:
        if a.max() <= 1.5:
            a = (a * 255.0).clip(0, 255)
        a = a.clip(0, 255).astype(np.uint8)
    if a.shape[-1] == 1:
        a = np.repeat(a, 3, axis=-1)
    return a



def main():
    parser = argparse.ArgumentParser()

    # Model parameters (match leaf script)
    parser.add_argument('--model_type', default='VGG')
    parser.add_argument('--pretrained', action='store_true')
    parser.add_argument('--loading_epoch', type=int, required=True)
    parser.add_argument('--timestamp', required=True)
    parser.add_argument('--outdim', type=int, default=2,
                        help='1=binary single-logit head, 2=softmax 2-class, 3=multiclass, dual_head uses 2 logits with sigmoid')
    parser.add_argument('--dual_head', action='store_true')
    parser.add_argument('--model_path', type=str, required=True)

    # Device
    parser.add_argument('--mps', action='store_true')
    parser.add_argument('--cuda', action='store_true')
    parser.add_argument('--cuda_id', default="0")

    # Patch source
    parser.add_argument('--patch_source', type=str, required=True,
                        help='Folder of patch images OR .hdf5 file containing patches')

    # NEW: Mask folder (binary masks, same 224x224, matched by filename stem)
    parser.add_argument('--mask_dir', type=str, default=None,
                        help='Folder containing binary masks for patches (matched by stem). If omitted, IoU is skipped.')
    parser.add_argument('--save_gt_overlay', action='store_true',
                        help='If set, saves GT-only overlay previews (useful sanity check).')

    # Thresholds (match leaf)
    parser.add_argument('--up_threshold', type=float, default=0.6)
    parser.add_argument('--down_threshold', type=float, default=0.2)
    parser.add_argument('--sal_threshold', type=float, default=0.7)
    parser.add_argument('--threshold', nargs='+', help='pixel SR thresholds (optional)')

    # Dual-head knobs (match leaf)
    parser.add_argument('--spor_th', type=float, default=None)
    parser.add_argument('--inf_gate', type=float, default=None)
    parser.add_argument('--dpi', type=int, required=False, default=0)

    # Saliency toggles (match leaf)
    parser.add_argument('--sal_gradcam', action='store_true')
    parser.add_argument('--sal_gradient', action='store_true')
    parser.add_argument('--sal_smoothgrad', action='store_true')
    parser.add_argument('--sal_deeplift', action='store_true')
    parser.add_argument('--sal_thresh_method', type=str, default='fixed',
                        choices=['percentile', 'fixed'])
    parser.add_argument('--sal_thresh_p', type=float, default=95.0)

    # Output
    parser.add_argument('--output_dir', type=str, default=None,
                        help='Defaults to <patch_source>/results/<model_string>/')
    parser.add_argument('--save_only_positive', action='store_true',
                        help='If set, only save overlays for infected/spor patches')
    parser.add_argument('--log', type=str, default='../../results/logs/random.log')

    # Normalization
    parser.add_argument('--means', type=float, nargs='+', default=[0.504, 0.604, 0.361])
    parser.add_argument('--stds', type=float, nargs='+', default=[0.144, 0.142, 0.192])

    parser.add_argument('--sweep', action='store_true',
                        help='If set, sweep thresholds and compute metric per head.')
    parser.add_argument('--sweep_mode', type=str, default='percentile',
                        choices=['percentile', 'fixed'],
                        help='Sweep percentiles (recommended) or fixed numeric thresholds in [0,1].')
    parser.add_argument('--sweep_percentiles', type=float, nargs='+',
                        default=[50, 60, 70, 80, 85, 90, 92, 94, 95, 96, 97, 98, 99],
                        help='Percentiles for saliency thresholding (keep top (100-p)%%).')
    parser.add_argument('--sweep_fixed', type=float, nargs='+',
                        default=[0.05, 0.10, 0.15, 0.20, 0.25, 0.30, 0.40, 0.50, 0.60, 0.70, 0.80, 0.90],
                        help='Fixed thresholds for heat in [0,1] after normalization.')
    parser.add_argument('--pick_metric', type=str, default='iou',
                        choices=['iou', 'dice'],
                        help='Metric to maximize when selecting best threshold.')

    opt = parser.parse_args()

    # device
    if opt.cuda:
        os.environ["CUDA_VISIBLE_DEVICES"] = str(opt.cuda_id)
        device_type = 'cuda'
    elif opt.mps:
        device_type = 'mps'
    else:
        device_type = 'cpu'

    logger = set_logging(Path(str(opt.log)), 20)
    logger.info(os.path.basename(__file__))
    printArgs(logger, vars(opt))

    model_root = Path(opt.model_path).expanduser().resolve()
    ref_dataset_path = model_root / "train_1_26_26_3_class_powdery.hdf5"

    model_para = parse_model(opt)
    model, device = load_model(model_para)
    model.eval()

    last_conv_layer = get_last_conv_layer(model)
    first_conv_layer = get_first_conv_layer(model)

    means = opt.means
    stds = opt.stds

    # preprocess
    if opt.model_type == 'Inception3':
        preprocess = tvtrans.Compose([
            tvtrans.ToPILImage(),
            tvtrans.Resize(299),
            tvtrans.ToTensor(),
            tvtrans.Normalize(means, stds),
        ])
        image_width = image_height = 299
    else:
        preprocess = tvtrans.Compose([
            tvtrans.ToPILImage(),
            tvtrans.ToTensor(),
            tvtrans.Normalize(means, stds),
        ])
        image_width = image_height = 224

    saliency_methods = get_saliency_methods(
        model,
        last_conv_layer=last_conv_layer,
        first_conv_layer=first_conv_layer,
        ref_dataset_path=ref_dataset_path,
        image_width=image_width,
        transform=preprocess,
        device=device,
        partial=True,
        explanation_map=False,
        gradcam=opt.sal_gradcam,
        gradient=opt.sal_gradient,
        smooth_grad=opt.sal_smoothgrad,
        deeplift=opt.sal_deeplift,
    )

    logger.info("Saliency methods enabled: %s",
                list(saliency_methods.keys()) if hasattr(saliency_methods, "keys") else saliency_methods)
    logger.info("Num saliency methods: %d", len(saliency_methods))

    default_cmap = LinearSegmentedColormap.from_list(
        "teal_purple_white",
        [
            (0.0, 1.0, 1.0),  # cyan
            (1.0, 1.0, 1.0),  # white
            (1.0, 0.0, 1.0),  # magenta
        ]
    )

    #default_cmap = LinearSegmentedColormap.from_list('MyColor', ['green', 'white', 'red'])
    fmt = "png"

    up_th = float(opt.up_threshold)
    down_th = float(opt.down_threshold)
    overlay_thresh_fixed = float(opt.sal_threshold)
    spor_th = float(opt.spor_th) if opt.spor_th is not None else up_th
    inf_gate = float(opt.inf_gate) if opt.inf_gate is not None else down_th

    model_string = f"{opt.model_type}_upth{up_th}_downth{down_th}_{opt.timestamp}"
    patch_source = Path(opt.patch_source)
    mask_dir = Path(opt.mask_dir) if opt.mask_dir else None

    if opt.output_dir:
        output_root = Path(opt.output_dir)
    else:
        output_root = patch_source.parent / "results" / model_string / patch_source.stem

    output_root.mkdir(parents=True, exist_ok=True)
    (output_root / "overlays").mkdir(exist_ok=True)
    (output_root / "bins").mkdir(exist_ok=True)
    (output_root / "gt_overlays").mkdir(exist_ok=True)

    pixel_th = [float(x) for x in opt.threshold] if opt.threshold else []

    patch_rows = []
    iou_rows = []  # NEW: long-format IoU rows
    summary = defaultdict(int)
    iou_summary = defaultdict(int)  # counts of patches with GT found, etc.

    if patch_source.suffix.lower() in (".h5", ".hdf5"):
        iterator = iter_patch_images_from_hdf5(patch_source)
    else:
        iterator = iter_patch_images_from_folder(patch_source)

    for name, payload in iterator:
        start_time = time.time()
        date_time_str = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

        # load patch -> uint8 HWC
        if isinstance(payload, (str, Path)):
            img = Image.open(payload).convert("RGB").resize((image_width, image_height))
            img_arr = np.asarray(img)
        else:
            img_arr = to_uint8_hwc(payload)
            img = Image.fromarray(img_arr).resize((image_width, image_height))
            img_arr = np.asarray(img)

        # Load GT mask if available
        gt_mask = None
        gt_path = None
        if mask_dir is not None:
            gt_path = find_mask_for_image(name, mask_dir)
            if gt_path is not None:
                gt_mask = load_binary_mask(gt_path, size=(image_width, image_height))
                iou_summary["gt_found"] += 1
                # optional sanity overlay
                if opt.save_gt_overlay:
                    base = (img_arr.astype(np.float32) / 255.0)
                    alpha = 0.45
                    out_fp = output_root / "gt_overlays" / f"{Path(name).stem}_GT.png"
                    alphas = np.full(gt_mask.shape, alpha, dtype=float)
                    alphas[~gt_mask] = 0.0
                    plt.imshow(base)
                    plt.imshow(gt_mask.astype(np.uint8), alpha=alphas, cmap=default_cmap)
                    plt.axis("off")
                    plt.tight_layout()
                    plt.savefig(out_fp, dpi=300, bbox_inches="tight", pad_inches=0)
                    plt.close()
            else:
                iou_summary["gt_missing"] += 1

        # model forward
        input_img = preprocess(img_arr).unsqueeze(0).to(device)
        input_img.requires_grad_(True)

        label = "discard"
        target_head = 0
        p_inf = None
        p_spor = None
        pred_class = None

        if opt.dual_head:
            logits = model(input_img)  # (1,2)
            probs = torch.sigmoid(logits).detach()
            p_inf = float(probs[0, 0].cpu().item())
            p_spor = float(probs[0, 1].cpu().item())

            if opt.dpi >= 5:
                if (p_inf > inf_gate) and (p_spor >= spor_th):
                    label = "spor"
                    target_head = 1
                elif p_inf >= up_th:
                    label = "infected"
                    target_head = 0
                elif p_inf <= down_th:
                    label = "clear"
                else:
                    label = "discard"
            else:
                if p_inf >= up_th:
                    label = "infected"
                    target_head = 0
                elif (p_inf > inf_gate) and (p_spor >= spor_th):
                    label = "spor"
                    target_head = 1
                elif p_inf <= down_th:
                    label = "clear"
                else:
                    label = "discard"

        else:
            logits = model(input_img)
            if opt.outdim == 1:
                p_inf = float(torch.sigmoid(logits)[0, 0].detach().cpu().item())
            else:
                prob = torch.softmax(logits, dim=1).detach()
                if prob.shape[1] >= 2:
                    p_inf = float(prob[0, 1].cpu().item())
                pred_class = int(torch.argmax(prob, dim=1)[0].cpu().item())

            if p_inf >= up_th:
                label = "infected"
                target_head = 0
            elif p_inf <= down_th:
                label = "clear"
            else:
                label = "discard"

        summary[label] += 1
        time_elapsed = timeSince(start_time)

        per_method_bin = {}
        per_method_thresh = {}

        per_method_bin_by_head = {}
        per_method_thresh_by_head = {}

        # ------------------------------------------------------
        # Saliency + IoU computation hooks
        # ------------------------------------------------------
        def record_iou_row(head_name: str, method: str, thr: float, pred_bin_u8: np.ndarray):
            """Append long-format IoU row if GT exists."""
            if gt_mask is None:
                return
            m = bin_metrics(pred_bin_u8.astype(bool), gt_mask)
            iou_rows.append({
                "timestamp": date_time_str,
                "filename": str(name),
                "gt_mask_path": str(gt_path) if gt_path is not None else "",
                "label": label,
                "head": head_name,
                "method": method,
                "sal_threshold": float(thr),
                "p_inf": p_inf,
                "p_spor": p_spor,
                **m
            })

        if opt.dual_head and len(saliency_methods) > 0:
            pos_inf = (p_inf is not None) and (p_inf >= up_th)
            pos_spor = (p_inf is not None) and (p_spor is not None) and (p_inf > inf_gate) and (p_spor >= spor_th)

            heads_to_explain = []

            if pos_inf:
                heads_to_explain.append((0, "infected_head"))
            if pos_spor:
                heads_to_explain.append((1, "spor_head"))

            for head_idx, head_name in heads_to_explain:
                output_masks = get_saliency_masks(
                    saliency_methods, input_img, head_idx, relu_attributions=True
                )
                abs_norm, _, _ = normalize_image_attr(img_arr, output_masks, hist=False)
                abs_norm.pop("Original", None)

                per_method_bin_by_head[head_idx] = {}
                per_method_thresh_by_head[head_idx] = {}

                for k, heat in abs_norm.items():
                    # baseline threshold (your existing behavior)
                    t0 = adaptive_threshold(
                        heat,
                        mask=None,
                        method=opt.sal_thresh_method,
                        p=float(opt.sal_thresh_p)
                    ) if opt.sal_thresh_method != "fixed" else overlay_thresh_fixed

                    t = float(t0)
                    thr_source = "baseline"

                    # optional sweep: choose t that maximizes IoU/Dice vs GT for THIS patch/method
                    if opt.sweep and (gt_mask is not None):
                        best, all_rows = sweep_thresholds(
                            heat=heat,
                            gt_mask=gt_mask,
                            mode=opt.sweep_mode,
                            percentiles=list(opt.sweep_percentiles),
                            fixed=list(opt.sweep_fixed),
                            pick_metric=opt.pick_metric
                        )

                        # record sweep curve rows
                        for r in all_rows:
                            iou_rows.append({
                                "timestamp": date_time_str,
                                "filename": str(name),
                                "gt_mask_path": str(gt_path) if gt_path is not None else "",
                                "label": label,
                                "head": head_name,  # ✅ FIXED: dual-head uses the real head
                                "method": k,
                                "source": "sweep",
                                "sweep_mode": r["sweep_mode"],
                                "p": r["p"],
                                "sal_threshold": r["thr"],
                                "p_inf": p_inf,
                                "p_spor": p_spor,
                                "iou": r["iou"],
                                "dice": r["dice"],
                                "precision": r["precision"],
                                "recall": r["recall"],
                                "pred_area": r["pred_area"],
                                "gt_area": r["gt_area"],
                                "inter": r["inter"],
                                "union": r["union"],
                                "is_best": (r["thr"] == best["thr"]),
                            })

                        t = float(best["thr"])
                        thr_source = "sweep_best"

                    # IMPORTANT: compute pred_bin AFTER choosing t
                    pred_bin = (heat >= t).astype(np.uint8)

                    per_method_thresh_by_head[head_idx][k] = float(t)
                    per_method_bin_by_head[head_idx][k] = pred_bin


                    # record a single "chosen threshold" row (baseline or sweep_best)
                    if gt_mask is not None:
                        m = bin_metrics(pred_bin, gt_mask)
                        iou_rows.append({
                            "timestamp": date_time_str,
                            "filename": str(name),
                            "gt_mask_path": str(gt_path) if gt_path is not None else "",
                            "label": label,
                            "head": head_name,
                            "method": k,
                            "source": thr_source,
                            "sweep_mode": (opt.sweep_mode if opt.sweep else ""),
                            "p": np.nan,
                            "sal_threshold": float(t),
                            "p_inf": p_inf,
                            "p_spor": p_spor,
                            "is_best": True,
                            **m
                        })

                    # save per-head overlay/bin using the FINAL pred_bin
                    alpha = 0.5
                    base = (img_arr.astype(np.float32) / 255.0)

                    cmap = CMAP_HYPHAE if head_idx == 0 else CMAP_SPOR

                    out_fp = output_root / "overlays" / f"{Path(name).stem}_{head_name}_{k}_th{t:.4f}.{fmt}"
                    alphas = np.full(pred_bin.shape, alpha, dtype=float)
                    alphas[pred_bin == 0] = 0.0
                    plt.imshow(base)
                    plt.imshow(pred_bin, alpha=alphas, cmap=cmap)
                    plt.axis("off")
                    plt.tight_layout()
                    plt.savefig(out_fp, format=fmt, dpi=300, bbox_inches="tight", pad_inches=0)
                    plt.close()

                    out_fp = output_root / "bins" / f"{Path(name).stem}_{head_name}_{k}_bin_th{t:.4f}.{fmt}"
                    plt.imshow(pred_bin, cmap=cmap)
                    plt.axis("off")
                    plt.tight_layout()
                    plt.savefig(out_fp, format=fmt, dpi=300, bbox_inches="tight", pad_inches=0)
                    plt.close()

                if not pixel_th and per_method_thresh_by_head[head_idx]:
                    driver_key = "GradCAM" if "GradCAM" in per_method_thresh_by_head[head_idx] else next(
                        iter(per_method_thresh_by_head[head_idx]))
                    pixel_th = [float(per_method_thresh_by_head[head_idx][driver_key])]

            # Optional: union IoU for dual-head if both computed (does saliency hit GT if either head highlights it?)
            if gt_mask is not None and (0 in per_method_bin_by_head) and (1 in per_method_bin_by_head):
                common_methods = sorted(set(per_method_bin_by_head[0]).intersection(per_method_bin_by_head[1]))
                for method in common_methods:
                    union_pred = np.logical_or(per_method_bin_by_head[0][method].astype(bool),
                                               per_method_bin_by_head[1][method].astype(bool)).astype(np.uint8)
                    # use "union" pseudo-head label
                    t0 = per_method_thresh_by_head[0][method]
                    t1 = per_method_thresh_by_head[1][method]
                    record_iou_row(head_name="union_heads", method=method, thr=float(np.nanmean([t0, t1])),
                                   pred_bin_u8=union_pred)

        elif (not opt.dual_head) and label in ("infected", "spor") and len(saliency_methods) > 0:
            output_masks = get_saliency_masks(
                saliency_methods, input_img, target_head, relu_attributions=True
            )
            abs_norm, _, _ = normalize_image_attr(img_arr, output_masks, hist=False)
            abs_norm.pop("Original", None)

            for k, heat in abs_norm.items():
                t0 = adaptive_threshold(
                    heat,
                    mask=None,
                    method=opt.sal_thresh_method,
                    p=float(opt.sal_thresh_p)
                ) if opt.sal_thresh_method != "fixed" else overlay_thresh_fixed

                t = float(t0)
                thr_source = "baseline"

                if opt.sweep and (gt_mask is not None):
                    best, all_rows = sweep_thresholds(
                        heat=heat,
                        gt_mask=gt_mask,
                        mode=opt.sweep_mode,
                        percentiles=list(opt.sweep_percentiles),
                        fixed=list(opt.sweep_fixed),
                        pick_metric=opt.pick_metric
                    )

                    for r in all_rows:
                        iou_rows.append({
                            "timestamp": date_time_str,
                            "filename": str(name),
                            "gt_mask_path": str(gt_path) if gt_path is not None else "",
                            "label": label,
                            "head": "single_head",
                            "method": k,
                            "source": "sweep",
                            "sweep_mode": r["sweep_mode"],
                            "p": r["p"],
                            "sal_threshold": r["thr"],
                            "p_inf": p_inf,
                            "p_spor": p_spor,
                            "iou": r["iou"],
                            "dice": r["dice"],
                            "precision": r["precision"],
                            "recall": r["recall"],
                            "pred_area": r["pred_area"],
                            "gt_area": r["gt_area"],
                            "inter": r["inter"],
                            "union": r["union"],
                            "is_best": (r["thr"] == best["thr"]),
                        })

                    t = float(best["thr"])
                    thr_source = "sweep_best"

                pred_bin = (heat >= t).astype(np.uint8)

                per_method_thresh[k] = float(t)
                per_method_bin[k] = pred_bin

                if gt_mask is not None:
                    m = bin_metrics(pred_bin, gt_mask)
                    iou_rows.append({
                        "timestamp": date_time_str,
                        "filename": str(name),
                        "gt_mask_path": str(gt_path) if gt_path is not None else "",
                        "label": label,
                        "head": "single_head",
                        "method": k,
                        "source": thr_source,
                        "sweep_mode": (opt.sweep_mode if opt.sweep else ""),
                        "p": np.nan,
                        "sal_threshold": float(t),
                        "p_inf": p_inf,
                        "p_spor": p_spor,
                        "is_best": True,
                        **m
                    })

                if (not opt.save_only_positive) or (label in ("infected", "spor")):
                    alpha = 0.5
                    base = (img_arr.astype(np.float32) / 255.0)

                    out_fp = output_root / "overlays" / f"{Path(name).stem}_{label}_{k}_th{t:.4f}.{fmt}"
                    alphas = np.full(pred_bin.shape, alpha, dtype=float)
                    alphas[pred_bin == 0] = 0.0
                    plt.imshow(base)
                    plt.imshow(pred_bin, alpha=alphas, cmap=default_cmap)
                    plt.axis("off")
                    plt.tight_layout()
                    plt.savefig(out_fp, format=fmt, dpi=300, bbox_inches="tight", pad_inches=0)
                    plt.close()

                    out_fp = output_root / "bins" / f"{Path(name).stem}_{label}_{k}_bin_th{t:.4f}.{fmt}"
                    plt.imshow(pred_bin, cmap=default_cmap)
                    plt.axis("off")
                    plt.tight_layout()
                    plt.savefig(out_fp, format=fmt, dpi=300, bbox_inches="tight", pad_inches=0)
                    plt.close()

            if not pixel_th and per_method_thresh:
                driver_key = "GradCAM" if "GradCAM" in per_method_thresh else next(iter(per_method_thresh))
                pixel_th = [float(per_method_thresh[driver_key])]

        # severity metrics (unchanged)
        severity_rate_patch = None
        severity_rates_pixel = {}

        patch_info = {
            "infected_patch": summary["infected"] + summary["spor"],
            "conidiophore_patch": summary["spor"],
            "clear_patch": summary["clear"],
            "discard_patch": summary["discard"],
            "lost_focus_patch": 0,
        }

        heatmap_info = {}
        if per_method_bin:
            first_k = next(iter(per_method_bin))
            heatmap_info[first_k] = per_method_bin[first_k].astype(np.float32)
        heatmap_info["prob_heatmap1"] = np.full((IMG_HEIGHT, IMG_WIDTH), float(p_inf if p_inf is not None else 0.0),
                                                dtype=np.float32)

        threshold_info = {
            "patch_down_th": down_th,
            "patch_up_th": up_th,
            "pixel_th": pixel_th if pixel_th else [overlay_thresh_fixed],
        }

        try:
            if opt.dual_head:
                severity_rate_patch, _ = patch_sr.metric_two_class(patch_info, heatmap_info, threshold_info)
            else:
                severity_rate_patch, _ = patch_sr.metric(patch_info, heatmap_info, threshold_info)
        except Exception:
            severity_rate_patch = None

        try:
            severity_rates_pixel, _ = pixel_sr1.metric(
                patch_info.copy(), heatmap_info.copy(), threshold_info.copy(), opt.outdim
            )
        except Exception:
            severity_rates_pixel = {}

        patch_rows.append({
            "timestamp": date_time_str,
            "filename": str(name),
            "label": label,
            "p_inf": p_inf,
            "p_spor": p_spor,
            "pred_class": pred_class,
            "up_th": up_th,
            "down_th": down_th,
            "spor_th": spor_th,
            "inf_gate": inf_gate,
            "sal_threshold_fixed": overlay_thresh_fixed,
            "sal_thresh_method": opt.sal_thresh_method,
            "sal_thresh_p": float(opt.sal_thresh_p),
            "severity_rate_patch_running": severity_rate_patch,
            "gt_found": bool(gt_mask is not None),
            "gt_path": str(gt_path) if gt_path is not None else "",
            "time_elapsed": time_elapsed,
        })

        logger.info("Processed %s label=%s p_inf=%s p_spor=%s in %s",
                    name, label, f"{p_inf:.4f}" if p_inf is not None else "NA",
                    f"{p_spor:.4f}" if p_spor is not None else "NA",
                    time_elapsed)

    # write CSVs
    df = pd.DataFrame(patch_rows)
    df.to_csv(output_root / "patch_results.csv", index=False)

    # NEW: IoU long-format results

    summary_row = {
        "total_patches": int(len(df)),
        "clear_patches": int(summary["clear"]),
        "infected_patches": int(summary["infected"]),
        "spor_patches": int(summary["spor"]),
        "discard_patches": int(summary["discard"]),
        "gt_found": int(iou_summary.get("gt_found", 0)),
        "gt_missing": int(iou_summary.get("gt_missing", 0)),
        "model_string": model_string,
        "patch_source": str(patch_source),
        "mask_dir": str(mask_dir) if mask_dir else "",
    }
    pd.DataFrame([summary_row]).to_csv(output_root / "summary.csv", index=False)

    if len(iou_rows) > 0:
        iou_df = pd.DataFrame(iou_rows)
        iou_df.to_csv(output_root / "iou_results_long.csv", index=False)

        if opt.sweep:
            iou_df = pd.DataFrame(iou_rows)

            # only sweep rows (not the sweep_best / baseline summaries)
            sweep_df = iou_df[iou_df.get("source", "") == "sweep"].copy()

            if len(sweep_df) == 0:
                logger.warning("No sweep rows found; sweep_curve.csv will be empty.")
            else:
                if opt.sweep_mode == "percentile":
                    group_cols = ["head", "method", "sweep_mode", "p"]  # p is defined here
                else:
                    group_cols = ["head", "method", "sweep_mode", "sal_threshold"]  # p is NA, don't group on it

                best_global = (sweep_df
                               .groupby(group_cols, as_index=False)
                               .agg(mean_iou=("iou", "mean"),
                                    median_iou=("iou", "median"),
                                    n=("iou", "size"))
                               .sort_values(["head", "method", "mean_iou"], ascending=[True, True, False])
                               )

                # top threshold per head/method
                top = best_global.groupby(["head", "method"], as_index=False).head(1)

                top.to_csv(output_root / "iou_best_thresholds.csv", index=False)
                best_global.to_csv(output_root / "sweep_curve.csv", index=False)

    logger.info("Saved outputs to %s", output_root)


if __name__ == "__main__":
    main()
