To run inference on an image date, drop the date folder with nested tray folder here. Due to GitHub size limitations, we have just included [one example](https://github.com/mswiseman/mildewVision2/tree/main/data/6-28-2023_10dpi/1) where `--imaging_date` would be 6-28-2023_10dpi, `--dpi` would be the 10, and `--tray` would be 1. Leaf disks used for the human testing are available on Zenodo. 

An example command might like like this: 

```
time python code/leaf_correlation_mw.py \
    # Model
    --model_type ResNet \
    --model_path ../.. \
    --loading_epoch 44 \
    --timestamp Jan26_23-15-35_2026 \
    --pretrained \
    --dual_head \

    # Data
    --dataset_path ../../data \
    --img_folder 6-28-2023_10dpi \
    --dpi 10 \
    --trays 1 \

    # Thresholds / inference logic
    --up_threshold 0.95 \
    --down_threshold 0.3 \
    --inf_gate 0.3 \
    --spor_th 0.5 \

    # Model outputs
    --outdim 2 \

    # Normalization
    --means 0.5663 0.6596 0.4508 \
    --stds 0.1811 0.1667 0.2434 \

    # Saliency
    --sal_thresh_method fixed \
    --sal_gradcam \
    --sal_smoothgrad \

    # Hardware
    --cuda \
    --cuda_id 0
