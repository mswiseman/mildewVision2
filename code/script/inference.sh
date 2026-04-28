#!/bin/bash

#python ../classification/inference.py                                           \
#            --model_path     ../..                                              \
#            --dataset_path   ../..                                              \
#            --model_type ResNet                                                 \
#            --HDF5                                                              \
#            --amp                                                               \
#            --pretrained                                                        \
#            --set test                                                           \
#            --cuda                                                              \
#            --cuda_id 0                                                         \
#            --loading_epoch 30                                                   \
#            --means         0.5466 0.6427 0.4240                                \
#            --stds          0.1771 0.1648 0.2370                                \
#            --dual_head                                                         \
#            --timestamp Jan29_22-02-46_2026                                      \
#            --rescue_gate 0.55                                                  \
#            --spor_th 0.6                                                       \
#            --up_threshold 0.63                                                 \
#            --down_threshold 0.1                                               \
#            --n_misclassified 20                                               \
#            --save_misclassified \
#            --grid_search
#
python ../classification/inference.py                                           \
            --model_path     ../..                                              \
            --dataset_path   ../..                                              \
            --model_type ResNet                                                 \
            --HDF5                                                              \
            --amp                                                               \
            --pretrained                                                        \
            --set test                                                           \
            --cuda                                                              \
            --cuda_id 0                                                         \
            --loading_epoch 41                                                  \
            --means 0.5663 0.6596 0.4508                                        \
            --dual_head                                                         \
            --stds 0.1811 0.1667 0.2434                                         \
            --up_threshold 0.95                                                  \
            --down_threshold 0.3                                               \
            --timestamp Jan29_15-54-22_2026                                     \
            --inf_gate 0.3                                                     \
            --spor_th 0.5  \
            --ignore_discard


python ../classification/inference.py                                           \
            --model_path     ../..                                              \
            --dataset_path   ../..                                              \
            --model_type ResNet                                                 \
            --HDF5                                                              \
            --amp                                                               \
            --pretrained                                                        \
            --set test                                                           \
            --cuda                                                              \
            --cuda_id 0                                                         \
            --loading_epoch 34                                                  \
            --means 0.5663 0.6596 0.4508                                        \
            --dual_head                                                         \
            --stds 0.1811 0.1667 0.2434                                         \
            --up_threshold 0.95                                                  \
            --down_threshold 0.3                                               \
            --timestamp Jan29_17-39-29_2026                                     \
            --inf_gate 0.3                                                     \
            --spor_th 0.5  \
            --ignore_discard

python ../classification/inference.py                                           \
            --model_path     ../..                                              \
            --dataset_path   ../..                                              \
            --model_type ResNet                                                 \
            --HDF5                                                              \
            --amp                                                               \
            --pretrained                                                        \
            --set test                                                           \
            --cuda                                                              \
            --cuda_id 0                                                         \
            --loading_epoch 42                                                  \
            --means 0.5663 0.6596 0.4508                                        \
            --dual_head                                                         \
            --stds 0.1811 0.1667 0.2434                                         \
            --up_threshold 0.95                                                  \
            --down_threshold 0.3                                               \
            --timestamp Jan29_19-16-09_2026                                     \
            --inf_gate 0.3                                                     \
            --spor_th 0.5  \
            --ignore_discard

python ../classification/inference.py                                           \
            --model_path     ../..                                              \
            --dataset_path   ../..                                              \
            --model_type ResNet                                                 \
            --HDF5                                                              \
            --amp                                                               \
            --pretrained                                                        \
            --set test                                                           \
            --cuda                                                              \
            --cuda_id 0                                                         \
            --loading_epoch 53                                                  \
            --means 0.5663 0.6596 0.4508                                        \
            --dual_head                                                         \
            --stds 0.1811 0.1667 0.2434                                         \
            --up_threshold 0.95                                                  \
            --down_threshold 0.3                                               \
            --timestamp Jan29_21-19-46_2026                                     \
            --inf_gate 0.3                                                     \
            --spor_th 0.5  \
            --ignore_discard

python ../classification/inference.py                                           \
            --model_path     ../..                                              \
            --dataset_path   ../..                                              \
            --model_type ResNet                                                 \
            --HDF5                                                              \
            --amp                                                               \
            --pretrained                                                        \
            --set test                                                           \
            --cuda                                                              \
            --cuda_id 0                                                         \
            --loading_epoch 41                                                  \
            --means 0.5663 0.6596 0.4508                                        \
            --dual_head                                                         \
            --stds 0.1811 0.1667 0.2434                                         \
            --up_threshold 0.95                                                  \
            --down_threshold 0.3                                               \
            --timestamp Feb05_14-47-06_2026                                     \
            --inf_gate 0.2                                                     \
            --spor_th 0.5  \
            --ignore_discard

# For Jan26/29 models:
#--means 0.5466 0.6427 0.4240 \
#--stds 0.1771 0.1648 0.2370   \

# For Feb model:
#  --means 0.5410 0.6371 0.4188 \
#  --stds 0.1764 0.1650 0.2326 \