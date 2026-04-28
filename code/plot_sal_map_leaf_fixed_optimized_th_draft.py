import os
import time
import argparse
import pandas as pd
import numpy as np
from datetime import datetime
import warnings
import gc
from collections import defaultdict
from PIL import Image, ImageTk
from pathlib import Path
from matplotlib import pyplot as plt

import torch
import torch.nn.functional as F
from torchvision import transforms as tvtrans

from analyzer_config import (CHANNELS, IMG_HEIGHT, IMG_WIDTH, IMG_EXT, INPUT_SIZE)

from metric import pixel_sr1, patch_sr

from classification.inference import pred_img, refine_patch_maps
from classification.utils import timeSince, printArgs, load_model, parse_model, set_logging, adaptive_threshold, \
    save_small_rgb

from analysis.leaf_mask import leaf_mask, on_focus

from visualization.viz_util import _normalize_image_attr
from visualization.viz_helper import (get_first_conv_layer, get_last_conv_layer, viz_image_attr, normalize_image_attr,
                                      plot_figs, save_figs, overlay_two_binary_masks)

from sanity_check.utils import get_saliency_methods, get_saliency_masks

from scipy import ndimage
import matplotlib
from matplotlib.colors import LinearSegmentedColormap

matplotlib.use("Agg")

gc.collect()
torch.cuda.empty_cache()

np.random.seed(2020)

""" Usage
Analyze the full-size leaf disc images and calculate the severity rate
Given a date, do analysis on all the data collected in that date
"""

parser = argparse.ArgumentParser()

# Model parameters
parser.add_argument('--model_type', default='VGG', help='model used for training')
parser.add_argument('--pretrained', action='store_true', help='use pretrained model parameters')
parser.add_argument('--loading_epoch', type=int, required=True, help='xth model loaded for inference')
parser.add_argument('--timestamp', required=True, help='model timestamp')
parser.add_argument('--outdim', type=int, default=2,
                    help='number of model outputs: 1=binary infected head, 2=dual-head (infected + sporulating)')
parser.add_argument('--dual_head', action='store_true', help='Use a model with a dual head.')
parser.add_argument('--model_path', type=str, required=True, help='root path to the model')
parser.add_argument('--step_size', type=int, default=224, help='step size of sliding window')
parser.add_argument('--means', type=float, nargs='+', default=[0.504, 0.604, 0.361])
parser.add_argument('--stds', type=float, nargs='+', default=[0.144, 0.142, 0.192])
parser.add_argument(
    '--target_class', type=int, default=1,
    help='saliency target head')
parser.add_argument('--contam_control', action='store_true', help='use contamination control conditional logic')
parser.add_argument('--spor_th', type=float, default=None,
                    help='sporulation threshold for dual-head (defaults to up_threshold if not set)')
parser.add_argument('--pm', type=str, help='PM isolate used for inoculation - collected for metadata in the csv')
parser.add_argument('--inf_gate', type=float, default=None,
                    help='minimum infected prob required to allow sporulation call (defaults to down_threshold)')
parser.add_argument(
    '--store_both_sal_heads',
    action='store_true',
    help='For dual-head models, store saliency for both infected and spor heads when each head passes its threshold, rather than only for the final exclusive patch label.'
)

# CPU/GPU/MSP parameters
parser.add_argument('--mps', action='store_true', help='enable mps')
parser.add_argument('--cuda', action='store_true', help='enable cuda')
parser.add_argument('--cuda_id', default="0", help='specify cuda id')

# Output parameters
parser.add_argument('--save_infected', action='store_true', help='save infected images')
parser.add_argument('--save_conidiophores', action='store_true', help='save conidiophores images')
parser.add_argument('--save_healthy', action='store_true', help='save healthy images')
parser.add_argument('--sal_threshold', type=float, default=0.7, help='threshold for saliency map')
parser.add_argument('--save_discarded', action='store_true', help='save discarded images')

# Data analysis parameters
parser.add_argument('--up_threshold', type=float, default=0.6, help='upper threshold for severity ratio')
parser.add_argument('--down_threshold', type=float, default=0.2, help='lower threshold for severity ratio')
parser.add_argument('--dataset_path', type=str, required=True, help='root path to the data')
parser.add_argument('--fill_gaps', action='store_true', help='apply spatial heuristic to fill gaps in predictions')
parser.add_argument('--img_folder', type=str, default="2-5-2023_6dpi", help='directory of images')
parser.add_argument('--platform', type=str, default='BlackBird', help='robot platform (Pmbot or BlackBird)')
parser.add_argument('--threshold', nargs='+', help='thresholding value for pixel sr')
parser.add_argument('--log', type=str, default='../../results/logs/random.log', help='log file path')
parser.add_argument('--dpi', type=int, required=True, help='inoculation date')
parser.add_argument('--group', type=str, default='baseline', help='exp group')
parser.add_argument('--trays', nargs='+', help='trays')

# saliency mapping flags
parser.add_argument('--sal_gradcam', action='store_true')
parser.add_argument('--sal_gradient', action='store_true')
parser.add_argument('--sal_smoothgrad', action='store_true')
parser.add_argument('--sal_deeplift', action='store_true')
parser.add_argument('--sal_thresh_method', type=str, default='fixed',
                    choices=['percentile', 'fixed'],
                    help='How to compute saliency threshold per image/method')
parser.add_argument('--sal_thresh_p', type=float, default=95.0,
                    help='Percentile used when method=percentile')

# output format
parser.add_argument("--out_format", default="jpg", choices=["png", "jpg", "webp"])
parser.add_argument("--out_quality", type=int, default=85)  # jpg/webp quality
parser.add_argument("--out_max_side", type=int, default=1600)  # resize cap (px)

opt = parser.parse_args()

# ------------------------------------------------------------
# Optimized fixed saliency thresholds (dual-head aware)
#   - Used when --sal_thresh_method fixed
#   - Keys are (head_name, canonical_method_name) -> threshold
# ------------------------------------------------------------
HEAD_NAME = {0: "infected_head", 1: "spor_head"}

FIXED_SAL_THRESH = {
    ("infected_head", "DeepLift"): 0.4,
    ("infected_head", "GradCAM"): 0.7,
    ("infected_head", "Gradient-SG"): 0.5,

    ("spor_head", "DeepLift"): 0.1,
    ("spor_head", "GradCAM"): 0.2,
    ("spor_head", "Gradient-SG"): 0.1,
}


def canon_method_name(k: str) -> str:
    """Map your internal saliency method keys to the names used in FIXED_SAL_THRESH."""
    s = str(k).strip().lower()
    if "gradcam" in s:
        return "GradCAM"
    if "deeplift" in s:
        return "DeepLift"
    # your flag is --sal_smoothgrad; your table uses Gradient-SG
    if "smooth" in s or "sg" in s:
        return "Gradient-SG"
    return str(k)


def fixed_threshold_for(head_idx: int, method_key: str, default: float) -> float:
    head = HEAD_NAME.get(int(head_idx), "infected_head")
    m = canon_method_name(method_key)
    return float(FIXED_SAL_THRESH.get((head, m), default))


def safe_divide_heatmap(num: np.ndarray, den: np.ndarray) -> np.ndarray:
    """Divide num/den only where den>0; elsewhere 0 (prevents NaNs)."""
    out = np.zeros_like(num, dtype=np.float32)
    np.divide(num, den, out=out, where=(den > 0))
    return out


# filter out routine warnings
warnings.filterwarnings("ignore", category=UserWarning, module="captum.attr._core.deep_lift")

# set device
if opt.cuda:
    os.environ["CUDA_VISIBLE_DEVICES"] = str(opt.cuda_id)
    device_type = 'cuda'
elif opt.mps:
    device_type = 'mps'
else:
    device_type = 'cpu'

# set logging options
logger = set_logging(Path(str(opt.log)), 20)
logger.info(os.path.basename(__file__))
printArgs(logger, vars(opt))

# set paths
ref_dataset_path = {
    'root_path': Path(opt.dataset_path),
    'train_filepath': Path(opt.dataset_path) / 'train.hdf5',
    'test_filepath': Path(opt.dataset_path) / 'test.hdf5',
}
image_timestamp = opt.img_folder
model_timestamp = opt.timestamp
model_type = opt.model_type

PM = opt.pm
dual_head = opt.dual_head
outdim = opt.outdim
dataset_path = Path(opt.dataset_path) / image_timestamp
mask_path = Path(opt.dataset_path) / f'{image_timestamp}_masking'
model_string = model_type + '_upth' + str(opt.up_threshold) + '_downth' + str(
    opt.down_threshold) + '_' + opt.timestamp
output_folder = Path(opt.dataset_path).parents[0] / 'results' / model_string / image_timestamp

# Threshold for severity ratio
down_th = opt.down_threshold  # below this will be classified as healthy
up_th = opt.up_threshold  # above this will be classified as infected or conidiophores
pixel_th = opt.threshold if opt.threshold else []
spor_th = opt.spor_th if opt.spor_th is not None else opt.up_threshold
inf_gate = opt.inf_gate if opt.inf_gate is not None else opt.down_threshold
overlay_thresh_fixed = float(opt.sal_threshold)

rel_th = 0.2 # leaf mask; raise to crop more
target_class = int(opt.target_class) if opt.target_class != 'None' else None
step_size = opt.step_size

# Model
model_para = parse_model(opt)
model, device = load_model(model_para)
model.eval()
last_conv_layer = get_last_conv_layer(model)
first_conv_layer = get_first_conv_layer(model)

# Normalization
means = opt.means
stds = opt.stds

# Input preprocessing transformation
if opt.model_type == 'Inception3':
    preprocess = tvtrans.Compose([
        tvtrans.ToPILImage(),
        tvtrans.Resize(299),
        tvtrans.ToTensor(),
        tvtrans.Normalize(means, stds)
    ])
    image_width = image_height = 299
else:
    preprocess = tvtrans.Compose([
        tvtrans.ToPILImage(),
        # tvtrans.Lambda(lambda img: tvtrans.functional.adjust_brightness(img, 0.75)), # changed to improve conidiophore detection
        tvtrans.ToTensor(),
        tvtrans.Normalize(means, stds)
    ])
    image_width = image_height = 224

# Captum
saliency_methods = get_saliency_methods(model,
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
                                        deeplift=opt.sal_deeplift)

# Write severity ratio as CSV files

# --- CSV schema (match leaf_correlation_mw.py) ---
sal_method_keys = list(saliency_methods.keys())

META_COL_NAMES = ['timestamp', 'time_elapsed', 'model_type', 'model_timestamp', 'classes', 'imaging_date', 'tray',
                  'filename', 'conserved_identifier', 'USDA_number', 'CHUM_number_if_from_NCGR', 'other_name', 'PM',
                  'infected_threshold', 'healthy_threshold', 'sal_threshold'] + \
                 (['inf_gate', 'spor_th'] if dual_head else []) + \
                 ['leaf_mask_th', 'clear_patches', 'hyphal_patches'] + \
                 (['conidiophore_patches', 'sporulating_pct'] if dual_head else []) + \
                 ['discarded_patches', 'severity_rate_patch']

INF_TH_COLS = [f"{k}_inf_th" for k in sal_method_keys]
INF_SR_COLS = [f"{k}_inf_sr" for k in sal_method_keys]
SPOR_TH_COLS = [f"{k}_spor_th" for k in sal_method_keys]
SPOR_SR_COLS = [f"{k}_spor_sr" for k in sal_method_keys]
UNION_SR_COLS = [f"{k}_union_sr" for k in sal_method_keys]
INTER_SR_COLS = [f"{k}_intersect_sr" for k in sal_method_keys]

META_COL_NAMES = META_COL_NAMES + INF_TH_COLS + INF_SR_COLS + SPOR_TH_COLS + SPOR_SR_COLS + UNION_SR_COLS + INTER_SR_COLS

threshold = 0.7  # threshold for saliency map
# default_cmap = 'Blues'
default_cmap = LinearSegmentedColormap.from_list(
    'MyColor', ['green', 'white', 'red']
)

# Time
total_time = 0
total_time_2 = 0
format_ = 'png'

# Loop trays
for tray_id in opt.trays:
    dataset_tray_path = dataset_path / Path(tray_id)
    leaf_disk_image_filenames = [x for x in os.listdir(dataset_tray_path) if x.lower().endswith('.png')]

    # One DF per tray (match leaf_correlation_mw.py output schema)
    tray_run_stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    severity_rate_df = pd.DataFrame(columns=META_COL_NAMES)

    for leaf_disk_image_filename in leaf_disk_image_filenames:
        # Per-image timer + timestamp
        start_time = time.time()
        date_time_str = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        imagename_text = os.path.splitext(leaf_disk_image_filename)[0]

        logger.info('-------------------------------------------')
        logger.info('Processing %s tray=%s file=%s', image_timestamp, tray_id, leaf_disk_image_filename)

        # Load image
        img_filepath = dataset_tray_path / leaf_disk_image_filename
        img = Image.open(img_filepath)
        img_arr = np.asarray(img)
        width, height = img.size

        # Sliding geometry
        subim_x = (width - IMG_WIDTH) // step_size + 1
        subim_y = (height - IMG_HEIGHT) // step_size + 1
        subim_height = (subim_y - 1) * step_size + IMG_HEIGHT
        subim_width = (subim_x - 1) * step_size + IMG_WIDTH
        sub_img = img.crop((0, 0, subim_width, subim_height))
        sub_img_arr = np.asarray(sub_img)

        # Masking
        imask = leaf_mask(img, rel_th=rel_th)
        if imask is None:
            logger.info('Image: %s\tmasking ERROR', imagename_text)
            continue
        imask = (imask.astype('uint8') // 255)  # 0/1

        t1 = time.time()
        logger.info('Finished loading mask: %s', timeSince(start_time))

        # Per-image counters and buffers
        patch_idx = coor_x = coor_y = 0
        infected_patch = conidiophore_patch = clear_patch = discard_patch = lost_focus_patch = 0

        counting_map = np.zeros((height, width), dtype=np.float32)

        # Head-specific counting maps for saliency normalization (avoid dilution by clear/discard patches)
        counting_map_inf = np.zeros((height, width), dtype=np.float32)
        counting_map_spor = np.zeros((height, width), dtype=np.float32) if dual_head else None
        prob_attrs1 = np.zeros((subim_x * subim_y, IMG_HEIGHT, IMG_WIDTH), dtype=np.float32)
        if dual_head:
            prob_attrs2 = np.zeros((subim_x * subim_y, IMG_HEIGHT, IMG_WIDTH), dtype=np.float32)

        saliency_attrs_inf = {k: np.zeros((subim_x * subim_y, IMG_HEIGHT, IMG_WIDTH), dtype=np.float32)
                              for k in saliency_methods.keys()}
        saliency_attrs_spor = {k: np.zeros((subim_x * subim_y, IMG_HEIGHT, IMG_WIDTH), dtype=np.float32)
                               for k in saliency_methods.keys()} if dual_head else None

        # Output folder for this image
        f = imagename_text
        output_leaf_disk_image_folder = output_folder / f'{opt.dpi}dpi_{tray_id}_{f}'
        os.makedirs(output_leaf_disk_image_folder, exist_ok=True)

        # -------------------------------
        # Patch loop
        # -------------------------------
        # ---- thresholds ----
        spor_th = float(opt.spor_th) if opt.spor_th is not None else float(up_th)
        inf_gate = float(opt.inf_gate) if opt.inf_gate is not None else float(down_th)

        # -------------------------------
        # Patch loop
        # -------------------------------
        for _ in range(subim_y):
            for _ in range(subim_x):
                subim_mask = imask[coor_y:coor_y + IMG_HEIGHT, coor_x:coor_x + IMG_WIDTH]

                if not on_focus(subim_mask):
                    lost_focus_patch += 1
                    prob_attrs1[patch_idx].fill(-np.inf)
                    if dual_head:
                        prob_attrs2[patch_idx].fill(-np.inf)

                    # still advance geometry
                    counting_map[coor_y:coor_y + IMG_HEIGHT, coor_x:coor_x + IMG_WIDTH] += 1.0
                    coor_x += step_size
                    patch_idx += 1
                    continue

                # Crop & preprocess
                box = (coor_x, coor_y, coor_x + IMG_WIDTH, coor_y + IMG_HEIGHT)
                subim = img.crop(box).resize((image_width, image_height))
                subim_arr = np.asarray(subim)
                need_sal = bool(saliency_methods)

                input_img = preprocess(subim_arr).unsqueeze(0).to(device)
                if need_sal:
                    input_img.requires_grad_(True)

                if dual_head:
                    # Model forward (multi-label heads) -> sigmoid probs
                    logits = model(input_img)  # (1,1) or (1,2)
                    probs = torch.sigmoid(logits).detach()  # (1,1) or (1,2)
                    p_inf = float(probs[0, 0].cpu().item())
                    p_spor = float(probs[0, 1].cpu().item())

                else:
                    logits = model(input_img)  # (1,2)
                    prob = torch.softmax(logits, dim=1)  # (1,2)
                    p_inf = float(prob[0, 1].item())  # infected prob

                # Save probs for reconstruction
                prob_attrs1[patch_idx] = p_inf
                if dual_head and outdim >= 2:
                    prob_attrs2[patch_idx] = p_spor

                # Decide label + saliency head
                patch_label = "discard"
                target_head = 0

                if not dual_head:
                    # binary: clear / infected / discard band
                    if p_inf >= up_th:
                        patch_label = "infected"
                        infected_patch += 1
                        target_head = 0
                    elif p_inf <= down_th:
                        patch_label = "clear"
                        clear_patch += 1
                    else:
                        discard_patch += 1

                else:
                    # dual-head: clear / infected / spor (conidiophore) / discard band
                    # keep your dpi-specific precedence
                    if opt.dpi >= 5:
                        # spor first (gated), then infected
                        if (p_inf > inf_gate) and (p_spor >= spor_th):
                            patch_label = "spor"
                            conidiophore_patch += 1
                            target_head = 1
                        elif p_inf >= up_th:
                            patch_label = "infected"
                            infected_patch += 1
                            target_head = 0
                        elif p_inf <= down_th:
                            patch_label = "clear"
                            clear_patch += 1
                        else:
                            discard_patch += 1
                    else:
                        # infected first, then spor (gated)
                        if p_inf >= up_th:
                            patch_label = "infected"
                            infected_patch += 1
                            target_head = 0
                        #elif (p_inf > inf_gate) and (p_spor >= spor_th):
                        #    patch_label = "spor"
                        #    conidiophore_patch += 1
                        #    target_head = 1
                        elif p_inf <= down_th:
                            patch_label = "clear"
                            clear_patch += 1
                        else:
                            discard_patch += 1

                # Saliency only for positive calls (infected or spor)
                if saliency_methods:
                    if not dual_head:
                        # keep existing single-head behavior
                        if patch_label == "infected":
                            output_masks = get_saliency_masks(
                                saliency_methods, input_img, 0, relu_attributions=True
                            )
                            abs_norm, _, _ = normalize_image_attr(subim_arr, output_masks, hist=False)
                            abs_norm.pop("Original", None)

                            for key, val in abs_norm.items():
                                if image_height != IMG_HEIGHT:
                                    v = torch.from_numpy(val[None, None, ...])
                                    v = F.interpolate(v, (IMG_HEIGHT, IMG_WIDTH), mode="nearest")[0, 0].numpy()
                                    saliency_attrs_inf[key][patch_idx] = v
                                else:
                                    saliency_attrs_inf[key][patch_idx] = val

                            counting_map_inf[coor_y:coor_y + IMG_HEIGHT, coor_x:coor_x + IMG_WIDTH] += 1.0

                    else:
                        # Decide which saliency heads to store
                        if getattr(opt, "store_both_sal_heads", False):
                            # New headwise behavior
                            run_inf_sal = (p_inf >= up_th)
                            run_spor_sal = (p_spor is not None) and (p_inf >= inf_gate) and (p_spor >= spor_th)
                        else:
                            # Old exclusive behavior
                            run_inf_sal = (patch_label == "infected")
                            run_spor_sal = (patch_label == "spor")

                        if run_inf_sal:
                            output_masks_inf = get_saliency_masks(
                                saliency_methods, input_img, 0, relu_attributions=True
                            )
                            abs_norm_inf, _, _ = normalize_image_attr(subim_arr, output_masks_inf, hist=False)
                            abs_norm_inf.pop("Original", None)

                            for key, val in abs_norm_inf.items():
                                if image_height != IMG_HEIGHT:
                                    v = torch.from_numpy(val[None, None, ...])
                                    v = F.interpolate(v, (IMG_HEIGHT, IMG_WIDTH), mode="nearest")[0, 0].numpy()
                                    saliency_attrs_inf[key][patch_idx] = v
                                else:
                                    saliency_attrs_inf[key][patch_idx] = val

                            counting_map_inf[coor_y:coor_y + IMG_HEIGHT, coor_x:coor_x + IMG_WIDTH] += 1.0

                        if run_spor_sal:
                            output_masks_spor = get_saliency_masks(
                                saliency_methods, input_img, 1, relu_attributions=True
                            )
                            abs_norm_spor, _, _ = normalize_image_attr(subim_arr, output_masks_spor, hist=False)
                            abs_norm_spor.pop("Original", None)

                            for key, val in abs_norm_spor.items():
                                if image_height != IMG_HEIGHT:
                                    v = torch.from_numpy(val[None, None, ...])
                                    v = F.interpolate(v, (IMG_HEIGHT, IMG_WIDTH), mode="nearest")[0, 0].numpy()
                                    saliency_attrs_spor[key][patch_idx] = v
                                else:
                                    saliency_attrs_spor[key][patch_idx] = val

                            counting_map_spor[coor_y:coor_y + IMG_HEIGHT, coor_x:coor_x + IMG_WIDTH] += 1.0

                # Optional saves (now correct comparisons)
                if opt.save_discarded and patch_label == "discard":
                    out_dir = output_leaf_disk_image_folder / "discarded"
                    out_dir.mkdir(parents=True, exist_ok=True)
                    plt.imsave(out_dir / f"{imagename_text}_patch_{patch_idx}_discard.{format_}",
                               subim_arr, cmap=default_cmap, format=format_, dpi=300)

                if opt.save_healthy and patch_label == "clear":
                    out_dir = output_leaf_disk_image_folder / "clear"
                    out_dir.mkdir(parents=True, exist_ok=True)
                    plt.imsave(out_dir / f"{imagename_text}_patch_{patch_idx}_clear.{format_}",
                               subim_arr, cmap=default_cmap, format=format_, dpi=300)

                if opt.save_infected and patch_label == "infected":
                    out_dir = output_leaf_disk_image_folder / "infected"
                    out_dir.mkdir(parents=True, exist_ok=True)
                    plt.imsave(out_dir / f"{imagename_text}_patch_{patch_idx}_infected.{format_}",
                               subim_arr, cmap=default_cmap, format=format_, dpi=300)

                if (outdim > 2 or dual_head) and opt.save_conidiophores and patch_label == "spor":
                    out_dir = output_leaf_disk_image_folder / "conidiophores"
                    out_dir.mkdir(parents=True, exist_ok=True)
                    plt.imsave(out_dir / f"{imagename_text}_patch_{patch_idx}_spor.{format_}",
                               subim_arr, cmap=default_cmap, format=format_, dpi=300)

                # advance geometry
                counting_map[coor_y:coor_y + IMG_HEIGHT, coor_x:coor_x + IMG_WIDTH] += 1.0
                coor_x += step_size
                patch_idx += 1

            coor_x = 0
            coor_y += step_size

        counting_map[counting_map == 0] = 1.0

        logger.info('Finished crop and inference: %s', timeSince(start_time))

        if opt.fill_gaps:
            # -------------------------------
            # Spatial heuristic refinement (PATCH GRID)
            # -------------------------------
            p_inf_grid = np.zeros((subim_y, subim_x), dtype=np.float32)
            p_spor_grid = np.zeros((subim_y, subim_x), dtype=np.float32) if dual_head else None
            valid_grid = np.zeros((subim_y, subim_x), dtype=bool)

            # IMPORTANT: init ONCE
            patch_class = np.zeros((subim_y, subim_x), dtype=np.uint8)
            # 0=none/discard, 1=infected, 2=spor (spor is also infected)

            # Fill grids
            patch_idx = 0
            for gy in range(subim_y):
                for gx in range(subim_x):
                    v = np.isfinite(prob_attrs1[patch_idx].flat[0])
                    valid_grid[gy, gx] = bool(v)
                    if v:
                        p_inf_grid[gy, gx] = float(prob_attrs1[patch_idx].flat[0])
                        if dual_head:
                            p_spor_grid[gy, gx] = float(prob_attrs2[patch_idx].flat[0])
                    patch_idx += 1

            infected_refined, clear_refined = refine_patch_maps(
                p_inf_grid, valid_grid, up_th, down_th,
                do_close=True, close_iters=1,
                do_open=True, open_iters=1
            )

            if dual_head:
                spor_raw = (p_spor_grid >= spor_th) & valid_grid & (p_inf_grid > inf_gate)
                spor_refined = ndimage.binary_closing(spor_raw, structure=np.ones((3, 3)), iterations=1)
                spor_refined &= valid_grid
                spor_refined &= infected_refined
            else:
                spor_refined = None

            # Overwrite per-patch maps + recompute counts
            patch_idx = 0
            infected_patch = conidiophore_patch = clear_patch = discard_patch = 0

            for gy in range(subim_y):
                for gx in range(subim_x):
                    if not valid_grid[gy, gx]:
                        patch_idx += 1
                        continue

                    if clear_refined[gy, gx]:
                        patch_class[gy, gx] = 0
                        clear_patch += 1
                        prob_attrs1[patch_idx].fill(p_inf_grid[gy, gx])
                        if dual_head:
                            prob_attrs2[patch_idx].fill(p_spor_grid[gy, gx])

                    elif infected_refined[gy, gx]:
                        infected_patch += 1

                        if dual_head and spor_refined[gy, gx]:
                            patch_class[gy, gx] = 2
                            conidiophore_patch += 1
                        else:
                            patch_class[gy, gx] = 1

                        prob_attrs1[patch_idx].fill(p_inf_grid[gy, gx])
                        if dual_head:
                            prob_attrs2[patch_idx].fill(p_spor_grid[gy, gx])

                    else:
                        patch_class[gy, gx] = 0
                        discard_patch += 1
                        prob_attrs1[patch_idx].fill(p_inf_grid[gy, gx])
                        if dual_head:
                            prob_attrs2[patch_idx].fill(p_spor_grid[gy, gx])

                    patch_idx += 1
            logger.info('Finished spatial refinement: %s', timeSince(start_time))
        # -------------------------------
        # Reconstruction to full image
        # -------------------------------
        prob_heatmap1 = np.zeros((height, width), dtype=np.float32)
        if dual_head:
            prob_heatmap2 = np.zeros((height, width), dtype=np.float32)
        saliency_heatmaps_inf = {k: np.zeros((height, width), dtype=np.float32) for k in saliency_methods.keys()}
        saliency_heatmaps_spor = {k: np.zeros((height, width), dtype=np.float32) for k in
                                  saliency_methods.keys()} if dual_head else None

        class_map = np.zeros((height, width), dtype=np.float32)
        patch_idx = coor_x = coor_y = 0
        for _ in range(subim_y):
            for _ in range(subim_x):
                prob_heatmap1[coor_y:coor_y + IMG_HEIGHT, coor_x:coor_x + IMG_WIDTH] += prob_attrs1[patch_idx]
                if dual_head:
                    prob_heatmap2[coor_y:coor_y + IMG_HEIGHT, coor_x:coor_x + IMG_WIDTH] += prob_attrs2[patch_idx]
                for k in saliency_methods.keys():
                    saliency_heatmaps_inf[k][coor_y:coor_y + IMG_HEIGHT, coor_x:coor_x + IMG_WIDTH] += \
                        saliency_attrs_inf[k][patch_idx]
                    if dual_head:
                        saliency_heatmaps_spor[k][coor_y:coor_y + IMG_HEIGHT, coor_x:coor_x + IMG_WIDTH] += \
                            saliency_attrs_spor[k][patch_idx]
                coor_x += step_size
                patch_idx += 1
            coor_x = 0
            coor_y += step_size

        # Normalize by coverage
        prob_heatmap1 /= counting_map
        if dual_head:
            prob_heatmap2 /= counting_map
        for k in saliency_heatmaps_inf:
            saliency_heatmaps_inf[k] = safe_divide_heatmap(saliency_heatmaps_inf[k], counting_map_inf)
        if dual_head:
            for k in saliency_heatmaps_spor:
                saliency_heatmaps_spor[k] = safe_divide_heatmap(saliency_heatmaps_spor[k], counting_map_spor)

        # -------------------------------
        # Adaptive thresholds (once per image)
        # -------------------------------
        th_method = getattr(opt, 'sal_thresh_method', 'percentile')
        th_p = float(getattr(opt, 'sal_thresh_p', 95.0))
        adaptive_th_inf = {
            k: (
                adaptive_threshold(v, mask=imask, method=th_method, p=th_p)
                if th_method != 'fixed'
                else fixed_threshold_for(0, k, default=overlay_thresh_fixed)
            )
            for k, v in saliency_heatmaps_inf.items()
        }

        if dual_head:
            adaptive_th_spor = {
                k: (
                    adaptive_threshold(v, mask=imask, method=th_method, p=th_p)
                    if th_method != 'fixed'
                    else fixed_threshold_for(1, k, default=overlay_thresh_fixed)
                )
                for k, v in saliency_heatmaps_spor.items()
            }

        # If user didn’t provide pixel thresholds, pick one from saliency adaptively
        if not pixel_th and adaptive_th_inf:
            driver_key = 'GradCAM' if 'GradCAM' in adaptive_th_inf else next(iter(adaptive_th_inf))
            pixel_th = [float(adaptive_th_inf[driver_key])]

        # -------------------------------
        # Severity metrics
        # -------------------------------
        patch_info = {
            'infected_patch': infected_patch,
            'conidiophore_patch': conidiophore_patch,
            'clear_patch': clear_patch,
            'discard_patch': discard_patch,
            'lost_focus_patch': lost_focus_patch,
        }

        if dual_head:
            saliency_union = {
                k: np.maximum(saliency_heatmaps_inf[k], saliency_heatmaps_spor[k])
                for k in saliency_methods.keys()
            }
            heatmap_info = saliency_union
        else:
            heatmap_info = saliency_heatmaps_inf.copy()

        heatmap_info['prob_heatmap1'] = prob_heatmap1
        if dual_head:
            heatmap_info['prob_heatmap2'] = prob_heatmap2

        threshold_info = {
            'patch_down_th': down_th,
            'patch_up_th': up_th,
            'pixel_th': [float(x) for x in pixel_th] if pixel_th else [overlay_thresh_fixed],
        }

        if dual_head:
            severity_rate_patch, _ = patch_sr.metric_two_class(patch_info, heatmap_info, threshold_info)
            severity_rates_pixel, _ = pixel_sr1.metric(patch_info.copy(), heatmap_info.copy(), threshold_info.copy(),
                                                       outdim)
        else:
            severity_rate_patch, _ = patch_sr.metric(patch_info, heatmap_info, threshold_info)
            severity_rates_pixel, _ = pixel_sr1.metric(patch_info.copy(), heatmap_info.copy(), threshold_info.copy(),
                                                       outdim)

        if opt.contam_control and opt.dpi > 6 and outdim >= 2 and conidiophore_patch < 2 and infected_patch > 10:
            infected_patch = "NA"
            conidiophore_patch = "NA"

        # -------------------------------
        # Visualizations
        #   - raw leaf
        #   - masked leaf
        #   - patch-based class overlays using SAME palette as pixel overlays:
        #       infected-only = cyan, spor-only = magenta, overlap = white
        # -------------------------------
        alpha = 0.5

        # raw
        out_fp = output_leaf_disk_image_folder / f'{opt.dpi}dpi_{f}_raw.{format_}'
        plt.imshow(img_arr)
        plt.axis('off')
        plt.tight_layout()
        plt.savefig(out_fp, format=format_, dpi=300, bbox_inches='tight', pad_inches=0)
        plt.close()

        # masked background (float [0,1])
        base = img_arr.copy()
        base[imask == 0] = 0
        base = base.astype(np.float32) / 255.0

        # save masked leaf
        # out_fp = output_leaf_disk_image_folder / f'{opt.dpi}dpi_{f}_masked.{format_}'
        # save_small_rgb(out_fp, base, fmt=opt.out_format, quality=opt.out_quality, max_side=opt.out_max_side)

        if not dual_head:
            # infected mask from prob heatmap
            inf_bin = (prob_heatmap1 >= up_th).astype(np.uint8)
            inf_bin[imask == 0] = 0

            # Use cyan only (spor mask = all zeros)
            colored = overlay_two_binary_masks(base, inf_bin, np.zeros_like(inf_bin), alpha=alpha)

            output_leaf_disk_image_filepath = output_leaf_disk_image_folder / f'{opt.dpi}dpi_{f}_patch_based_class1.{format_}'

            fig, ax = plt.subplots()
            ax.imshow(colored)
            ax.axis("off")

            y = 0.98
            dy = 0.06
            text_kwargs = dict(color="white", fontsize=8,
                               bbox=dict(facecolor="black", alpha=0.5),
                               transform=ax.transAxes, va="top")

            ax.text(0.02, y, f"Healthy Patches: {clear_patch}", **text_kwargs)
            ax.text(0.02, y - dy, f"Hyphal Patches: {infected_patch}", **text_kwargs)
            ax.text(0.02, y - 2 * dy, f"Discarded Patches: {discard_patch}", **text_kwargs)
            ax.text(0.02, y - 3 * dy, f"Infection Severity Rate: {severity_rate_patch}", **text_kwargs)

            plt.tight_layout()
            plt.savefig(output_leaf_disk_image_filepath, format=format_, dpi=300, bbox_inches='tight', pad_inches=0)
            plt.close()

        else:
            # dual-head: infected + sporulation patch masks
            inf_bin = (prob_heatmap1 >= up_th).astype(np.uint8)
            spor_bin = ((prob_heatmap2 >= spor_th) & (prob_heatmap1 > inf_gate)).astype(np.uint8)

            # keep overlays only on leaf
            inf_bin[imask == 0] = 0
            spor_bin[imask == 0] = 0

            colored = overlay_two_binary_masks(base, inf_bin, spor_bin, alpha=alpha)

            output_leaf_disk_image_filepath = output_leaf_disk_image_folder / f'{opt.dpi}dpi_{f}_patch_based_both_classes.{format_}'

            # counts (keep your logic)
            hyphal_only_patch = max(int(infected_patch) - int(conidiophore_patch), 0)
            total_infected_patch = int(infected_patch)

            fig, ax = plt.subplots()
            ax.imshow(colored)
            ax.axis("off")

            y = 0.98
            dy = 0.06
            text_kwargs = dict(color="white", fontsize=8,
                               bbox=dict(facecolor="black", alpha=0.5),
                               transform=ax.transAxes, va="top")

            # ax.text(0.02, y, f"Healthy Patches: {clear_patch}", **text_kwargs)
            # ax.text(0.02, y - dy, f"Hyphal Patches: {hyphal_only_patch}", **text_kwargs)
            # ax.text(0.02, y - 2 * dy, f"Conidiophore Patches: {conidiophore_patch}", **text_kwargs)
            # ax.text(0.02, y - 3 * dy, f"Total Infected Patches: {total_infected_patch}", **text_kwargs)
            # ax.text(0.02, y - 4 * dy, f"Discarded Patches: {discard_patch}", **text_kwargs)
            # ax.text(0.02, y - 5 * dy, f"Infection Severity Rate: {severity_rate_patch}", **text_kwargs)

            plt.tight_layout()
            plt.savefig(output_leaf_disk_image_filepath, format=format_, dpi=300, bbox_inches='tight', pad_inches=0)
            plt.close()

        for key in saliency_methods.keys():
            # per-head bin maps
            t_inf = float(adaptive_th_inf.get(key, overlay_thresh_fixed))
            inf_bin = (saliency_heatmaps_inf[key] >= t_inf).astype(np.uint8)

            if not dual_head:
                # single-head: you can keep your old bin overlay, or use cyan only
                out = overlay_two_binary_masks(base, inf_bin, np.zeros_like(inf_bin), alpha=alpha)
                out_fp = output_leaf_disk_image_folder / f'{opt.dpi}dpi_{key}_INF_th{t_inf:.4f}_{f}_blended.{format_}'
                save_small_rgb(out_fp, out, fmt=opt.out_format, quality=opt.out_quality, max_side=opt.out_max_side)

            else:
                t_sp = float(adaptive_th_spor.get(key, overlay_thresh_fixed))
                spor_bin = (saliency_heatmaps_spor[key] >= t_sp).astype(np.uint8)

                # combined colored overlay
                combined = overlay_two_binary_masks(base, inf_bin, spor_bin, alpha=alpha)
                out_fp = output_leaf_disk_image_folder / f'{opt.dpi}dpi_{key}_COMBINED_INFth{t_inf:.4f}_SPth{t_sp:.4f}_{f}.png'
                save_small_rgb(out_fp, combined, fmt=opt.out_format, quality=opt.out_quality, max_side=opt.out_max_side)

                # (optional) also save per-head overlays for debugging
                inf_only_img = overlay_two_binary_masks(base, inf_bin, np.zeros_like(inf_bin), alpha=alpha)
                spor_only_img = overlay_two_binary_masks(base, np.zeros_like(spor_bin), spor_bin, alpha=alpha)

                save_small_rgb(
                    output_leaf_disk_image_folder / f"{opt.dpi}dpi_{key}_INF_th{t_inf:.4f}_{f}",
                    inf_only_img,
                    fmt="jpg",
                    quality=85,
                    max_side=1600
                )

                save_small_rgb(
                    output_leaf_disk_image_folder / f"{opt.dpi}dpi_{key}_SPOR_th{t_sp:.4f}_{f}",
                    spor_only_img,
                    fmt="jpg",
                    quality=85,
                    max_side=1600
                )

        # -------------------------------
        # Compute saliency rate metrics (match leaf_correlation_mw.py)
        # -------------------------------
        focus_mask = (prob_heatmap1 != -np.inf)
        leaf_mask_bin = (imask > 0)
        valid_mask = leaf_mask_bin & focus_mask
        valid_pixels = int(valid_mask.sum())

        by_method = {}
        for k in sal_method_keys:
            th_inf = float(adaptive_th_inf.get(k, overlay_thresh_fixed))
            inf_bin = (saliency_heatmaps_inf[k] >= th_inf)

            if dual_head:
                th_sp = float(adaptive_th_spor.get(k, overlay_thresh_fixed))
                spor_bin = (saliency_heatmaps_spor[k] >= th_sp)
            else:
                th_sp = np.nan
                spor_bin = np.zeros_like(inf_bin, dtype=bool)

            union = inf_bin | spor_bin
            inter = inf_bin & spor_bin

            if valid_pixels > 0:
                inf_sr = float(inf_bin[valid_mask].mean() * 100.0)
                spor_sr = float(spor_bin[valid_mask].mean() * 100.0)
                union_sr = float(union[valid_mask].mean() * 100.0)
                inter_sr = float(inter[valid_mask].mean() * 100.0)
            else:
                inf_sr = spor_sr = union_sr = inter_sr = np.nan

            by_method[k] = {
                "inf_th": th_inf,
                "inf_sr": round(inf_sr, 2) if np.isfinite(inf_sr) else np.nan,
                "spor_th": float(th_sp) if np.isfinite(th_sp) else np.nan,
                "spor_sr": round(spor_sr, 2) if np.isfinite(spor_sr) else np.nan,
                "union_sr": round(union_sr, 2) if np.isfinite(union_sr) else np.nan,
                "intersect_sr": round(inter_sr, 2) if np.isfinite(inter_sr) else np.nan,
            }

        # -------------------------------
        # CSV row creation (match leaf_correlation_mw.py)
        # -------------------------------
        conserved_identifier = ''  # not available in this script
        sporulating_pct = None
        if dual_head:
            total_patches = clear_patch + infected_patch + conidiophore_patch
            sporulating_pct = (conidiophore_patch / total_patches * 100.0) if total_patches > 0 else np.nan

        record_data = [
            date_time_str,  # timestamp
            timeSince(start_time),  # time_elapsed
            model_type,  # model_type
            model_timestamp,  # model_timestamp
            outdim,  # classes
            image_timestamp,  # imaging_date
            tray_id,  # tray
            imagename_text,  # filename
            conserved_identifier,  # conserved_identifier
            '',  # USDA_number
            '',  # CHUM_number_if_from_NCGR
            '',  # other_name
            PM,  # PM
            up_th,  # infected_threshold
            down_th,  # healthy_threshold
            float(overlay_thresh_fixed),  # sal_threshold (legacy field; keep a constant)
        ]

        if dual_head:
            record_data += [inf_gate, spor_th]  # inf_gate, spor_th

        record_data += [
            rel_th,  # leaf_mask_th
            clear_patch,  # clear_patches
            infected_patch,  # hyphal_patches
        ]

        if dual_head:
            record_data += [
                conidiophore_patch,  # conidiophore_patches
                sporulating_pct,  # sporulating_pct
            ]

        record_data += [
            discard_patch,  # discarded_patches
            severity_rate_patch,  # severity_rate_patch
        ]

        # Per-method columns in fixed order
        for k in sal_method_keys:
            record_data.append(by_method[k]["inf_th"])
        for k in sal_method_keys:
            record_data.append(by_method[k]["inf_sr"])
        for k in sal_method_keys:
            record_data.append(by_method[k]["spor_th"])
        for k in sal_method_keys:
            record_data.append(by_method[k]["spor_sr"])
        for k in sal_method_keys:
            record_data.append(by_method[k]["union_sr"])
        for k in sal_method_keys:
            record_data.append(by_method[k]["intersect_sr"])

        severity_rate_df = pd.concat(
            [severity_rate_df, pd.DataFrame([record_data], columns=META_COL_NAMES)],
            ignore_index=True
        )
        logger.info('Analysis finished: %s', timeSince(start_time))
        logger.info('-------------------------------------------')

        # Free per-image memory
        del img, img_arr, sub_img, sub_img_arr, prob_heatmap1, saliency_heatmaps_inf
        if dual_head:
            del prob_heatmap2, saliency_heatmaps_spor
        plt.close("all")
        gc.collect()
    # -------------------------------
    # Save CSV for this tray (match leaf_correlation_mw.py)
    # -------------------------------
    output_csv_folder_th = output_folder / 'th'
    os.makedirs(output_csv_folder_th, exist_ok=True)

    if dual_head:
        out_csv = output_csv_folder_th / (
            f'severity_rate_tray_{tray_id}'
            f'_u{up_th}_d{down_th}_ig{inf_gate}_sp{spor_th}'
            f'_{tray_run_stamp}.csv'
        )
    else:
        out_csv = output_csv_folder_th / (
            f'severity_rate_tray_{tray_id}'
            f'_u{up_th}_d{down_th}'
            f'_{tray_run_stamp}.csv'
        )

    severity_rate_df.to_csv(out_csv, index=False)
    logger.info('Saved %s', out_csv)
