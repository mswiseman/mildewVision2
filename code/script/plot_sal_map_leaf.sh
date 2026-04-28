#!/bin/bash


time python ../plot_sal_map_leaf.py                                   \
      --model_type ResNet \
      --model_path ../.. \
      --dataset_path ../../data/human_disk_assessments \
      --loading_epoch 44 \
      --cuda \
      --cuda_id 0 \
      --outdim 2 \
      --up_threshold 0.95 \
      --down_threshold 0.3 \
      --sal_threshold 0.5 \
      --means 0.5410 0.6371 0.4188 \
      --stds 0.1764 0.1650 0.2326 \
      --timestamp Jan26_23-15-35_2026 \
      --dpi 10 \
      --pretrained \
      --sal_thresh_method fixed             \
      --img_folder 1-22-2026 \
      --trays 10dpi \
      --pm HPM-1269 \
      --dual_head \
      --spor_th 0.7 \
      --inf_gate 0.3 \
      --sal_gradcam \
      --sal_smoothgrad