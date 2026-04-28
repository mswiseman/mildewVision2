#!/bin/bash

#declare -a h=("hello" "world")
#declare -a code_paths=("`/Users/tim/BB_analysis/code/classification/run.py`" "`/home/tq42/BB_analysis/code/classification/run.py`")
#declare -a root_paths=("`/Users/tim/Documents/Cornell/CAIR/BlackBird/Data/Hyphal_2019`" "`/mnt/cornell/Data/tq42/Hyphal_2020`")

#for((i=0;i<5;i++))
#do
    python ../segmentation/run.py      \
                --root_path  /c/Users/Intel\ User/Desktop/blackbird_scripts \
                --model_type DeepLab            \
                --pretrained                \
                --save_model                \
                --weighted_loss             \
                --loading_epoch 0           \
                --total_epochs 95           \
                --cuda                      \
                --optimType Adam            \
                --lr 1e-4                   \
                --weight_decay 2e-4         \
                --bsize 32                  \
                --nworker 1                 \
                --cuda_device 0

     #           --cv                        \
#                --seg_idx $i                \
#done
