import sys
import numpy as np

sys.path.append('../../../Desktop/blackbird_scripts/code')

from utils import hard_thresholding, otsu_thresholding
from analyzer_config import IMG_HEIGHT, IMG_WIDTH


def metric(patch_info, heatmap_info, threshold_info, outdim):
    """
    Calculate pixel-level severity rate using saliency heatmaps.

    Supports:
      - threshold_info['pixel_th'] (list): apply same threshold to all methods
      - threshold_info['pixel_th_by_method'] (dict): apply a different threshold per method key
    """
    # Remove probability heatmaps if present
    heatmap_info.pop('prob_heatmap', None)
    heatmap_info.pop('prob_heatmap1', None)
    heatmap_info.pop('prob_heatmap2', None)

    # Debug
    print("pixel_sr1 heatmap keys:", list(heatmap_info.keys()))

    lost_focus_patch = patch_info['lost_focus_patch']
    lost_focus_pixel = lost_focus_patch * IMG_HEIGHT * IMG_WIDTH

    # Backward-compatible: list of thresholds (applied to all methods)
    pixel_th_list = threshold_info.get('pixel_th', [])

    # New: dict of per-method thresholds
    pixel_th_by_method = threshold_info.get('pixel_th_by_method', None)

    severity_rates = {}
    infected_pixels = {}
    total_pixels = {}

    # --- Case A: per-method thresholds (recommended for your use case) ---
    if isinstance(pixel_th_by_method, dict) and pixel_th_by_method:
        th_label = "by_method"  # single bucket label; change if you prefer
        severity_rates[th_label] = {}
        infected_pixels[th_label] = {}
        total_pixels[th_label] = {}

        for key, val in heatmap_info.items():
            th = pixel_th_by_method.get(key, None)
            if th is None:
                # fallback if a method key isn't in the dict
                th = float(pixel_th_list[0]) if pixel_th_list else 0.5

            # Allow 'otsu' as a per-method option too, if you ever use it
            if th == 'otsu':
                saliency_mask = otsu_thresholding(val, vmin=0, vmax=1)
            else:
                saliency_mask = hard_thresholding(val, float(th), vmin=0, vmax=1)

            total_pixel = saliency_mask.size
            clear_pixel = np.sum(saliency_mask == 0)
            infected_pixel = np.sum(saliency_mask == 1) + (np.sum(saliency_mask == 2) if outdim == 3 else 0)

            assert clear_pixel + infected_pixel == total_pixel

            denom = (total_pixel - lost_focus_pixel)
            severity_rates[th_label][key] = round(infected_pixel / denom * 100, 2) if denom > 0 else np.nan
            infected_pixels[th_label][key] = infected_pixel
            total_pixels[th_label][key] = denom

        # Return per-method threshold results
        return severity_rates, (infected_pixels, total_pixels)

    # --- Case B: original behavior (same threshold for all methods) ---
    for th in pixel_th_list:
        for key, val in heatmap_info.items():
            saliency_mask = otsu_thresholding(val, vmin=0, vmax=1) if th == 'otsu' else hard_thresholding(
                val, float(th), vmin=0, vmax=1
            )

            total_pixel = saliency_mask.size
            clear_pixel = np.sum(saliency_mask == 0)
            infected_pixel = np.sum(saliency_mask == 1) + (np.sum(saliency_mask == 2) if outdim == 3 else 0)

            assert clear_pixel + infected_pixel == total_pixel

            denom = (total_pixel - lost_focus_pixel)
            severity_rates.setdefault(th, {})[key] = round(infected_pixel / denom * 100, 2) if denom > 0 else np.nan
            infected_pixels.setdefault(th, {})[key] = infected_pixel
            total_pixels.setdefault(th, {})[key] = denom

    # IMPORTANT: return should be AFTER the loops (your original indentation returned too early)
    return severity_rates, (infected_pixels, total_pixels)
