#!/bin/bash

PATCH_DIR="/c/Users/Intel User/Desktop/blackbird_scripts/data/patches_for_class_visualization"
#MASK_DIR="/c/Users/Intel User/Desktop/blackbird_scripts/data/segmentation/segmentation_annotations/train/masks"   # <-- change to your actual mask folder

time python ../plot_sal_map_patch_seg_iou_draft.py \
  --model_type ResNet \
  --model_path "/c/Users/Intel User/Desktop/blackbird_scripts" \
  --patch_source "$PATCH_DIR" \
  --pretrained \
  --loading_epoch 44 \
  --cuda --cuda_id 0 \
  --up_threshold 0.95 \
  --down_threshold 0.3 \
  --sal_threshold 0.1 \
  --spor_th 0.7 \
  --dual_head \
  --inf_gate 0.3 \
  --sal_smoothgrad \
  --sal_gradcam \
  --sal_deeplift \
  --sal_thresh_method fixed \
  --outdim 2 \
  --dpi 8 \
  --means 0.5663 0.6596 0.4508 \
  --stds 0.1811 0.1667 0.2434 \
  --timestamp Jan26_23-15-35_2026

    #--sweep \
    #--sweep_mode fixed \
    #--sweep_fixed 0.1 0.15 0.2 0.25 0.30 \
    #--mask_dir "$MASK_DIR" \
    #--save_gt_overlay \
    #--pick_metric dice \