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
    get_first_conv_layer, get_last_conv_layer, normalize_image_attr
)
from sanity_check.utils import get_saliency_methods, get_saliency_masks

# optional (only needed if you want pixel SR like leaf script)
from metric import pixel_sr1, patch_sr

np.random.seed(2020)
warnings.filterwarnings("ignore", category=UserWarning, module="captum.attr._core.deep_lift")


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
    If your file uses different keys, adjust the candidates list.
    """
    import h5py  # local import so folder mode doesn’t require it

    h5_path = Path(h5_path)
    with h5py.File(h5_path, "r") as f:
        # try common dataset keys
        image_keys = ["images", "imgs", "X", "data"]
        name_keys = ["filenames", "names", "ids", "image_names"]

        img_ds = None
        for k in image_keys:
            if k in f:
                img_ds = f[k]
                break
        if img_ds is None:
            # fallback: first dataset that looks like images
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
            # decode name if present
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
        # CHW -> HWC
        a = np.transpose(a, (1, 2, 0))
    if a.dtype != np.uint8:
        # assume [0,1] or [0,255]
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

    # Normalization (leaf uses CLI-provided means/stds; patch script had hard-coded)
    parser.add_argument('--means', type=float, nargs='+', default=[0.504, 0.604, 0.361])
    parser.add_argument('--stds', type=float, nargs='+', default=[0.144, 0.142, 0.192])

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

    # preprocess (keep leaf behavior)
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

    # captum setup (match leaf’s “partial=True, explanation_map=False” pattern)
    # NOTE: get_saliency_methods signature in your project already supports these flags (as used in leaf script)
    saliency_methods = get_saliency_methods(
        model,
        last_conv_layer=last_conv_layer,
        first_conv_layer=first_conv_layer,
        ref_dataset_path=ref_dataset_path,  # patches only; leaf uses train/test paths for some methods
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

    default_cmap = LinearSegmentedColormap.from_list('MyColor', ['green', 'white', 'red'])
    fmt = "png"

    up_th = float(opt.up_threshold)
    down_th = float(opt.down_threshold)
    overlay_thresh_fixed = float(opt.sal_threshold)
    spor_th = float(opt.spor_th) if opt.spor_th is not None else up_th
    inf_gate = float(opt.inf_gate) if opt.inf_gate is not None else down_th

    # output layout similar to leaf script
    model_string = f"{opt.model_type}_upth{up_th}_downth{down_th}_{opt.timestamp}"
    patch_source = Path(opt.patch_source)
    if opt.output_dir:
        output_root = Path(opt.output_dir)
    else:
        # default next to source
        output_root = patch_source.parent / "results" / model_string / patch_source.stem

    output_root.mkdir(parents=True, exist_ok=True)
    (output_root / "overlays").mkdir(exist_ok=True)
    (output_root / "bins").mkdir(exist_ok=True)

    # pixel thresholds
    pixel_th = [float(x) for x in opt.threshold] if opt.threshold else []

    # per-patch rows (like leaf records per image)
    patch_rows = []
    summary = defaultdict(int)

    # choose iterator based on source
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

            # same decision structure as leaf (dpi gating preserved)
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
                # single-logit binary
                p_inf = float(torch.sigmoid(logits)[0, 0].detach().cpu().item())
            else:
                # softmax binary/multiclass; infected assumed class 1 for 2-class
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

        # saliency (only positive calls like leaf)
        per_method_bin = {}
        per_method_thresh = {}

        # saliency: optionally compute BOTH heads when positive, then also save a combined overlay
        per_method_bin_by_head = {}  # {head: {method: binmask}}
        per_method_thresh_by_head = {}  # {head: {method: thr}}

        if opt.dual_head and len(saliency_methods) > 0:
            # decide which heads are "positive enough" to explain
            pos_inf = (p_inf is not None) and (p_inf >= up_th)  # or: (p_inf > inf_gate)
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
                    t = adaptive_threshold(
                        heat,
                        mask=None,
                        method=opt.sal_thresh_method,
                        p=float(opt.sal_thresh_p)
                    ) if opt.sal_thresh_method != "fixed" else overlay_thresh_fixed

                    per_method_thresh_by_head[head_idx][k] = float(t)
                    per_method_bin_by_head[head_idx][k] = (heat >= t).astype(np.uint8)

                    # save per-head overlay
                    alpha = 0.5
                    base = (img_arr.astype(np.float32) / 255.0)

                    out_fp = output_root / "overlays" / f"{Path(name).stem}_{head_name}_{k}_th{t:.4f}.{fmt}"
                    alphas = np.full(per_method_bin_by_head[head_idx][k].shape, alpha, dtype=float)
                    alphas[per_method_bin_by_head[head_idx][k] == 0] = 0.0
                    plt.imshow(base)
                    plt.imshow(per_method_bin_by_head[head_idx][k], alpha=alphas, cmap=default_cmap)
                    plt.axis("off")
                    plt.tight_layout()
                    plt.savefig(out_fp, format=fmt, dpi=300, bbox_inches="tight", pad_inches=0)
                    plt.close()

                    # save per-head bin map
                    out_fp = output_root / "bins" / f"{Path(name).stem}_{head_name}_{k}_bin_th{t:.4f}.{fmt}"
                    plt.imshow(per_method_bin_by_head[head_idx][k], cmap=default_cmap)
                    plt.axis("off")
                    plt.tight_layout()
                    plt.savefig(out_fp, format=fmt, dpi=300, bbox_inches="tight", pad_inches=0)
                    plt.close()

                # if no pixel_th provided, pick a driver threshold like before (per-head)
                if not pixel_th and per_method_thresh_by_head[head_idx]:
                    driver_key = "GradCAM" if "GradCAM" in per_method_thresh_by_head[head_idx] else next(
                        iter(per_method_thresh_by_head[head_idx]))
                    pixel_th = [float(per_method_thresh_by_head[head_idx][driver_key])]

            # --- combined overlay (only if we computed BOTH heads) ---
            if (0 in per_method_bin_by_head) and (1 in per_method_bin_by_head):
                methods0 = set(per_method_bin_by_head[0].keys())
                methods1 = set(per_method_bin_by_head[1].keys())
                common_methods = sorted(methods0.intersection(methods1))

                for method in common_methods:
                    inf_bin = per_method_bin_by_head[0][method]
                    spor_bin = per_method_bin_by_head[1][method]

                    CYAN = np.array([0.0, 1.0, 1.0])
                    MAGENTA = np.array([1.0, 0.0, 1.0])
                    WHITE = np.array([1.0, 1.0, 1.0])

                    alpha = 0.5
                    base = (img_arr.astype(np.float32) / 255.0)

                    inf_only = (inf_bin == 1) & (spor_bin == 0)
                    spor_only = (spor_bin == 1) & (inf_bin == 0)
                    overlap = (inf_bin == 1) & (spor_bin == 1)

                    overlay = base.copy()

                    overlay[inf_only] = (1 - alpha) * overlay[inf_only] + alpha * CYAN
                    overlay[spor_only] = (1 - alpha) * overlay[spor_only] + alpha * MAGENTA
                    overlay[overlap] = (1 - alpha) * overlay[overlap] + alpha * WHITE

                    out_fp = output_root / "overlays" / f"{Path(name).stem}_COMBINED_{method}.png"
                    plt.imshow(overlay)
                    plt.axis("off")
                    plt.tight_layout()
                    plt.savefig(out_fp, format="png", dpi=300, bbox_inches="tight", pad_inches=0)
                    plt.close()

        elif (not opt.dual_head) and label in ("infected", "spor") and len(saliency_methods) > 0:
            output_masks = get_saliency_masks(
                saliency_methods, input_img, target_head, relu_attributions=True
            )
            abs_norm, _, _ = normalize_image_attr(img_arr, output_masks, hist=False)
            abs_norm.pop("Original", None)

            # compute per-method thresholds (fixed or percentile), like leaf
            for k, heat in abs_norm.items():
                t = adaptive_threshold(
                    heat,
                    mask=None,
                    method=opt.sal_thresh_method,
                    p=float(opt.sal_thresh_p)
                ) if opt.sal_thresh_method != "fixed" else overlay_thresh_fixed

                per_method_thresh[k] = float(t)
                per_method_bin[k] = (heat >= t).astype(np.uint8)

                # save outputs (unless user wants only summary)
                if (not opt.save_only_positive) or (label in ("infected", "spor")):
                    alpha = 0.5
                    base = (img_arr.astype(np.float32) / 255.0)

                    # overlay
                    out_fp = output_root / "overlays" / f"{Path(name).stem}_{label}_{k}_th{t:.4f}.{fmt}"
                    alphas = np.full(per_method_bin[k].shape, alpha, dtype=float)
                    alphas[per_method_bin[k] == 0] = 0.0
                    plt.imshow(base)
                    plt.imshow(per_method_bin[k], alpha=alphas, cmap=default_cmap)
                    plt.axis("off")
                    plt.tight_layout()
                    plt.savefig(out_fp, format=fmt, dpi=300, bbox_inches="tight", pad_inches=0)
                    plt.close()

                    # binary map
                    out_fp = output_root / "bins" / f"{Path(name).stem}_{label}_{k}_bin_th{t:.4f}.{fmt}"
                    plt.imshow(per_method_bin[k], cmap=default_cmap)
                    plt.axis("off")
                    plt.tight_layout()
                    plt.savefig(out_fp, format=fmt, dpi=300, bbox_inches="tight", pad_inches=0)
                    plt.close()

            # if no pixel_th provided, pick a driver threshold like leaf (GradCAM preferred)
            if not pixel_th and per_method_thresh:
                driver_key = "GradCAM" if "GradCAM" in per_method_thresh else next(iter(per_method_thresh))
                pixel_th = [float(per_method_thresh[driver_key])]

        # severity metrics (patch-level + optional pixel SR)
        # For patch-only mode: patch_sr reduces to counts; pixel_sr uses saliency bins if available.
        severity_rate_patch = None
        severity_rates_pixel = {}

        # patch_sr wants patch_info / heatmap_info / threshold_info (same structure leaf uses)
        patch_info = {
            "infected_patch": summary["infected"] + summary["spor"],
            "conidiophore_patch": summary["spor"],
            "clear_patch": summary["clear"],
            "discard_patch": summary["discard"],
            "lost_focus_patch": 0,
        }

        # make a dummy "heatmap" for pixel_sr based on one method (if present)
        heatmap_info = {}
        if per_method_bin:
            # pick first method as representative; pixel_sr expects float maps
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
            # keep going; patch-only runs may not have all metric expectations satisfied
            severity_rate_patch = None

        try:
            # pixel_sr1.metric returns dict keyed by threshold -> method values (project-specific)
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
            "time_elapsed": time_elapsed,
        })

        logger.info("Processed %s label=%s p_inf=%s p_spor=%s in %s",
                    name, label, f"{p_inf:.4f}" if p_inf is not None else "NA",
                    f"{p_spor:.4f}" if p_spor is not None else "NA",
                    time_elapsed)

    # write CSVs
    df = pd.DataFrame(patch_rows)
    df.to_csv(output_root / "patch_results.csv", index=False)

    summary_row = {
        "total_patches": int(len(df)),
        "clear_patches": int(summary["clear"]),
        "infected_patches": int(summary["infected"]),
        "spor_patches": int(summary["spor"]),
        "discard_patches": int(summary["discard"]),
        "model_string": model_string,
        "patch_source": str(patch_source),
    }
    pd.DataFrame([summary_row]).to_csv(output_root / "summary.csv", index=False)

    logger.info("Saved outputs to %s", output_root)


if __name__ == "__main__":
    main()
