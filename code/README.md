# Training

* For classification training, refer to the bash script: [script/train.sh](script/train.sh)
  
<details>
<summary><b>Example cross-validation classification train (click to expand)</b></summary>
  
  ```bash
  #!/bin/bash
for i in {0..4}
do
  echo "-----------------------------------------"
  echo "Starting fold $i"
  echo "-----------------------------------------"

  time python ../classification/run.py \
    --cuda_device   0 \
    --save_model \
    --root_path     ../.. \
    --model_type    ResNet \
    --test_hdf5     val_${i}.hdf5 \
    --train_hdf5    train_${i}.hdf5 \
    --pretrained \
    --weighted_loss \
    --cuda \
    --loading_epoch 0 \
    --total_epochs  100 \
    --outdim        2 \
    --optim_type    AdamW \
    --lr            1e-4 \
    --patience      25 \
    --weight_decay  1e-4 \
    --scheduler \
    --bsize         128 \
    --nworker       1 \
    --cross_validation \
    --test_date     Jan_26_2026 \
    --means         0.5466 0.6427 0.4240 \
    --stds          0.1771 0.1648 0.2370

  echo "Finished fold $i"
done

echo "All training runs completed."
```

</details>

* For segmentation training, refer to the bash script: [script/train_seg.sh](script/train_seg.sh)

# Inference

* [script/plot_sal_map_leaf.sh](script/plot_sal_map_leaf.sh): for classification inference returning flagged saliency maps

<details>
<summary><b>Example plot_sal_map_leaf.sh script</b></summary>

```bash
#!/bin/bash

time python ../plot_sal_map_leaf.py \
    # Model
    --model_type    ResNet \
    --model_path    ../.. \
    --loading_epoch 44 \
    --timestamp     Jan26_23-15-35_2026 \
    --pretrained \
    --dual_head \

    # Data
    --dataset_path  ../../data/human_disk_assessments \
    --img_folder    1-22-2026 \
    --trays         1 \
    --dpi           10 \
    --pm            HPM-1269 \

    # Thresholds / inference logic
    --up_threshold   0.95 \
    --down_threshold 0.3 \
    --inf_gate       0.3 \
    --spor_th        0.7 \

    # Saliency
    --sal_threshold       0.5 \
    --sal_thresh_method   fixed \
    --sal_gradcam \
    --sal_smoothgrad \

    # Normalization
    --means 0.5410 0.6371 0.4188 \
    --stds  0.1764 0.1650 0.2326 \

    # Hardware
    --cuda \
    --cuda_id 0 \

    # Output
    --outdim 2

```
</details>


* For parallel classification inference with or without saliency maps (customizable), refer to: [script/plot_leaf_correlation_all.sh](script/plot_leaf_correlation_all.sh)
* [plot_sal_map_leaf_fixed_optimized_th.py](plot_sal_map_leaf_fixed_optimized_th.py): This script acts like [plot_sal_map_leaf.py](plot_sal_map_leaf.py), but hardcodes the optimized thresholds determined by [plot_sal_map_seg_iou.py](plot_sal_map_seg_iou.py)

# Optimization 
* To check if saliency maps actually overlap the manually annotated mildew regions and return best thresholds, run this: refer to: [plot_sal_map_patch_seg_iou.py](plot_sal_map_patch_seg_iou.py)
<details>
<summary><b>Example IoU Optimization (click to expand)</b></summary>

```bash
#!/bin/bash

PATCH_DIR="data/patches_for_class_visualization"
MASK_DIR="data/segmentation/segmentation_annotations/train/masks"

time python ../plot_sal_map_patch_seg_iou_draft.py \
  --model_type ResNet \
  --model_path /c/Users/Intel\ User/Desktop/blackbird_scripts \
  --patch_source "$PATCH_DIR" \
  --pretrained \
  --loading_epoch 44 \
  --cuda --cuda_id 0 \
  \
  # thresholds
  --up_threshold 0.95 \
  --down_threshold 0.3 \
  --inf_gate 0.3 \
  --spor_th 0.7 \
  \
  # saliency
  --sal_threshold 0.1 \
  --sal_thresh_method fixed \
  --sal_smoothgrad \
  --sal_gradcam \
  --sal_deeplift \
  \
  # sweep
  --sweep \
  --sweep_mode fixed \
  --sweep_fixed 0.1 0.15 0.2 0.25 0.30 \
  --pick_metric dice \
  \
  # model setup
  --dual_head \
  --outdim 2 \
  --dpi 8 \
  \
  # normalization
  --means 0.5663 0.6596 0.4508 \
  --stds 0.1811 0.1667 0.2434 \
  \
  # metadata
  --timestamp Jan26_23-15-35_2026 \
  \
  # masks
  --mask_dir "$MASK_DIR" \
  --save_gt_overlay

</details>
