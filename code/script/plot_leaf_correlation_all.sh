#!/usr/bin/env bash

#
# -----------------------------------------------------------------------------
# Script: plot_leaf_correlation_all.sh
#
# Runs multiple leaf analysis pipelines in parallel using the active Python
# environment (expected: conda env "mildewVision"). Examples as written launch:
#   1) Disease severity and saliency map generation (plot_sal_map_leaf.py)
#   2) Disease severity and optional saliency analysis (leaf_correlation.py)
#
# For information on argparse arguments see argparse section in either 
# plot_sal_map_leaf.py or leaf_correlation.py
#
# Usage:
#   bash plot_leaf_correlation_all.sh
# -----------------------------------------------------------------------------

# Move to the directory this script lives in (no hard-coded Windows paths)
# On 309 computer that should be ~/Desktop/blackbird_scripts/
cd "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)" || {
    echo "Failed to cd into script directory" >&2
    exit 1
}

# Find python from the current environment (should be mildewVision)
PYTHON="$(command -v python || true)"
if [[ -z "$PYTHON" ]]; then
    echo "python not found on PATH. Activate your conda env first:" >&2
    echo "    conda activate mildewVision" >&2
    echo "then run:" >&2
    echo "    bash $(basename "$0")" >&2
    exit 1
fi

# Limit number of concurrent jobs
MAX_JOBS=2

commands=(
    "time \"$PYTHON\" ../plot_sal_map_leaf_fixed_optimized_th.py \
        --model_type ResNet \
        --model_path ../.. \
        --dataset_path ../../data \
        --loading_epoch 44 \
        --up_threshold 0.95 \
        --down_threshold 0.3 \
        --inf_gate 0.3 \
        --spor_th 0.5 \
        --cuda \
        --cuda_id 0 \
        --outdim 2 \
        --means 0.5663 0.6596 0.4508 \
        --stds 0.1811 0.1667 0.2434 \
        --timestamp Jan26_23-15-35_2026 \
        --dpi 10 \
        --pretrained \
        --sal_thresh_method fixed \
        --sal_smoothgrad \
        --sal_deeplift \
        --sal_gradcam \
        --dual_head \
        --img_folder 1-22-2026 \
        --trays 5dpi"

    "time \"$PYTHON\" ../plot_sal_map_leaf_fixed_optimized_th.py \
        --model_type ResNet \
        --model_path ../.. \
        --dataset_path ../../data \
        --loading_epoch 44 \
        --up_threshold 0.95 \
        --down_threshold 0.3 \
        --inf_gate 0.3 \
        --spor_th 0.5 \
        --cuda \
        --cuda_id 0 \
        --outdim 2 \
        --means 0.5663 0.6596 0.4508 \
        --stds 0.1811 0.1667 0.2434 \
        --timestamp Jan26_23-15-35_2026 \
        --dpi 10 \
        --pretrained \
        --sal_thresh_method fixed \
        --sal_smoothgrad \
        --sal_deeplift \
        --sal_gradcam \
        --dual_head \
        --img_folder 1-22-2026 \
        --trays 10dpi \
        --store_both_sal_heads"

    "time \"$PYTHON\" ../plot_sal_map_leaf_fixed_optimized_th_draft.py \
        --model_type ResNet \
        --model_path ../.. \
        --dataset_path ../../data \
        --loading_epoch 44 \
        --up_threshold 0.95 \
        --down_threshold 0.3 \
        --inf_gate 0.3 \
        --spor_th 0.5 \
        --cuda \
        --cuda_id 0 \
        --outdim 2 \
        --means 0.5663 0.6596 0.4508 \
        --stds 0.1811 0.1667 0.2434 \
        --timestamp Jan26_23-15-35_2026 \
        --dpi 10 \
        --pretrained \
        --sal_thresh_method fixed \
        --sal_smoothgrad \
        --sal_deeplift \
        --sal_gradcam \
        --dual_head \
        --img_folder 1-22-2026 \
        --trays 5dpi \
        --store_both_sal_heads"

    "time \"$PYTHON\" ../leaf_correlation_mw.py \
        --model_type ResNet \
        --model_path ../.. \
        --dataset_path ../../data \
        --loading_epoch 44 \
        --up_threshold 0.95 \
        --down_threshold 0.3 \
        --inf_gate 0.3 \
        --spor_th 0.5 \
        --cuda \
        --cuda_id 0 \
        --outdim 2 \
        --means 0.5663 0.6596 0.4508 \
        --stds 0.1811 0.1667 0.2434 \
        --timestamp Jan26_23-15-35_2026 \
        --dpi 10 \
        --pretrained \
        --sal_thresh_method fixed \
        --sal_smoothgrad \
        --sal_deeplift \
        --sal_gradcam \
        --dual_head \
        --img_folder 1-22-2026 \
        --trays 10dpi"

    "time \"$PYTHON\" ../leaf_correlation_mw.py \
        --model_type ResNet \
        --model_path ../.. \
        --dataset_path ../../data \
        --loading_epoch 44 \
        --up_threshold 0.95 \
        --down_threshold 0.3 \
        --inf_gate 0.3 \
        --spor_th 0.5 \
        --cuda \
        --cuda_id 0 \
        --outdim 2 \
        --means 0.5663 0.6596 0.4508 \
        --stds 0.1811 0.1667 0.2434 \
        --timestamp Jan26_23-15-35_2026 \
        --dpi 5 \
        --pretrained \
        --sal_thresh_method fixed \
        --sal_smoothgrad \
        --sal_deeplift \
        --sal_gradcam \
        --dual_head \
        --img_folder 1-22-2026 \
        --trays 5dpi"
)


wait_for_free_job_slot() {
    while : ; do
        jobs_running=$(jobs -r | wc -l)
        if [[ $jobs_running -lt $MAX_JOBS ]]; then
            break
        fi
        sleep 1
    done
}

# Launch commands, respecting MAX_JOBS
for cmd in "${commands[@]}"; do
    eval "$cmd" &
    wait_for_free_job_slot
done

# Wait for all background jobs to complete
wait
