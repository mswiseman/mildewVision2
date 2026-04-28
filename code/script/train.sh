#!/bin/bash

###############################################################################
# run.py training launcher
#
# This script trains classification model using
# ../classification/run.py from the Blackbird / mildewVision pipeline.
#
# To run:
#   bash train.sh
#
# ----------------------- Available Parameters ------------------------------
#
# ========================= MODEL PARAMETERS =========================
#
#   --model_type         Model architecture
#                        Choices:
#                        GoogleNet, ResNet, SqueezeNet, DenseNet,
#                        VGG, AlexNet, Inception3
#
#   --pretrained         Use ImageNet-pretrained model parameters
#   --feature_extract    Fine-tune last layer only (freeze backbone) dont use with dual-head
#   --resume             Resume training from saved checkpoint
#   --resume_timestamp   Timestamp identifier of run to resume
#   --loading_epoch      Epoch number to load when resuming
#   --total_epochs       Total training epochs
#   --outdim             Number of output classes
#   --save_model         Save trained model checkpoints
#   --cuda               Enable CUDA GPU training
#   --mps                Enable Apple Metal Performance Shaders
#   --means              Channel-wise RGB mean values (from training set)
#   --stds               Channel-wise RGB std dev values (from training set)
#   --patience           Early stopping patience (epochs without improvement)
#
# ========================= OPTIMIZER PARAMETERS =========================
#
#   --optim_type         Optimizer selection
#                        Choices: Adam, Adadelta, RMSprop, SGD
#   --lr                 Learning rate
#   --weight_decay       L2 regularization factor
#   --weighted_loss      Apply class-weighted loss balancing (dont use with dual-head)
#   --max_grad_norm      Gradient clipping L2 norm (None disables)
#   --label_smoothing    Label smoothing factor (0.0 disables)
#
# ========================= SCHEDULER PARAMETERS =========================
#
#   --scheduler          Enable learning rate scheduler
#   --step_size          Epoch interval for LR decay
#   --gamma              LR decay multiplier
#
# ========================= DATALOADER PARAMETERS =========================
#
#   --bsize              Batch size
#   --nworker            Number of data loader workers
#   --manual_seed        Random seed for reproducibility
#   --cuda_device        CUDA device ID to use
#   --root_path          Root data directory (REQUIRED)
#   --test_date          Date string for dataset filtering
#   --train_hdf5         Training HDF5 filename
#   --test_hdf5          Testing / validation HDF5 filename
#   --qtl_partition_idx  QTL partition index
#   --seg_idx            Segmentation index for cross-validation
#   --demo_dataset       Use balanced demo dataset
#   --seg_dataset        Use randomized segmentation dataset
#   --aug_dataset        Use augmented dataset
#   --cross_validation   Enable cross-validation training
#
# ========================= FORTUNA / OPTUNA =========================
#
#   --n_trials           Number of hyperparameter trials
#   --study_name         Experiment name for Optuna/Fortuna study
#
# ---------------------------------------------------------------------------
# For complete implementation details, see argparse section in:
# classification/run.py
###############################################################################

echo "Starting training run..."
#
#
#time python ../classification/run.py                    \
#      --cuda_device   0                                   \
#      --save_model                                        \
#      --root_path     ../..                               \
#      --model_type    ResNet                              \
#      --test_hdf5     val_1_26_26_3_class_powdery.hdf5                          \
#      --train_hdf5    train_1_26_26_3_class_powdery.hdf5                        \
#      --pretrained                                        \
#      --weighted_loss                                     \
#      --cuda                                              \
#      --loading_epoch 0                                   \
#      --total_epochs  100                                 \
#      --outdim        2                                   \
#      --optim_type    AdamW                               \
#      --lr            1e-4                                \
#      --patience      25                                  \
#      --weight_decay  1e-4                                \
#      --scheduler                                         \
#      --bsize         128                                 \
#      --nworker       1                                   \
#      --cross_validation                                  \
#      --test_date    Jan_26_2026                          \
#      --study_name    downy_trial_11_24_24                \
#      --n_trials      0                                   \
#      --means         0.5466 0.6427 0.4240                \
#      --stds          0.1771 0.1648 0.2370
#
#
##
#time python ../classification/run.py                    \
#    --cuda_device   0                                   \
#    --save_model                                        \
#    --root_path     ../..                               \
#    --model_type    ResNet                              \
#    --test_hdf5     val.hdf5                            \
#    --train_hdf5    train.hdf5                          \
#    --pretrained                                        \
#    --weighted_loss                                     \
#    --cuda                                              \
#    --loading_epoch 0                                   \
#    --total_epochs  100                                 \
#    --outdim        2                                   \
#    --optim_type    AdamW                               \
#    --lr            1e-4                                \
#    --patience      25                                  \
#    --weight_decay  1e-4                                \
#    --scheduler                                         \
#    --bsize         128                                 \
#    --nworker       1                                   \
#    --study_name    none                                \
#    --n_trials      0                                   \
#    --means         0.5466 0.6427 0.4240                \
#    --stds          0.1771 0.1648 0.2370
#
#
#time python ../classification/run.py                    \
#    --cuda_device   0                                   \
#    --save_model                                        \
#    --root_path     ../..                               \
#    --model_type    ResNet                              \
#    --test_hdf5     val_0.hdf5                          \
#    --train_hdf5    train_0.hdf5                        \
#    --pretrained                                        \
#    --weighted_loss                                     \
#    --cuda                                              \
#    --loading_epoch 0                                   \
#    --total_epochs  100                                 \
#    --outdim        2                                   \
#    --optim_type    AdamW                               \
#    --lr            1e-4                                \
#    --patience      25                                  \
#    --weight_decay  1e-4                                \
#    --scheduler                                         \
#    --bsize         128                                 \
#    --nworker       1                                   \
#    --cross_validation                                  \
#    --test_date    Jan_26_2026                          \
#    --study_name    downy_trial_11_24_24                \
#    --n_trials      0                                   \
#    --means         0.5466 0.6427 0.4240                \
#    --stds          0.1771 0.1648 0.2370
#
#time python ../classification/run.py                    \
#    --cuda_device   0                                   \
#    --save_model                                        \
#    --root_path     ../..                               \
#    --model_type    ResNet                              \
#    --test_hdf5     val_1.hdf5                          \
#    --train_hdf5    train_1.hdf5                        \
#    --pretrained                                        \
#    --weighted_loss                                     \
#    --cuda                                              \
#    --loading_epoch 0                                   \
#    --total_epochs  100                                 \
#    --outdim        2                                   \
#    --optim_type    AdamW                               \
#    --lr            1e-4                                \
#    --patience      25                                  \
#    --weight_decay  1e-4                                \
#    --scheduler                                         \
#    --bsize         128                                 \
#    --nworker       1                                   \
#    --cross_validation                                  \
#    --test_date    Jan_26_2026                          \
#    --study_name    downy_trial_11_24_24                \
#    --n_trials      0                                   \
#    --means         0.5466 0.6427 0.4240                \
#    --stds          0.1771 0.1648 0.2370
#
#time python ../classification/run.py                    \
#    --cuda_device   0                                   \
#    --save_model                                        \
#    --root_path     ../..                               \
#    --model_type    ResNet                              \
#    --test_hdf5     val_2.hdf5                          \
#    --train_hdf5    train_2.hdf5                        \
#    --pretrained                                        \
#    --weighted_loss                                     \
#    --cuda                                              \
#    --loading_epoch 0                                   \
#    --total_epochs  100                                 \
#    --outdim        2                                   \
#    --optim_type    AdamW                               \
#    --lr            1e-4                                \
#    --patience      25                                  \
#    --weight_decay  1e-4                                \
#    --scheduler                                         \
#    --bsize         128                                 \
#    --nworker       1                                   \
#    --cross_validation                                  \
#    --test_date    Jan_26_2026                          \
#    --study_name    downy_trial_11_24_24                \
#    --n_trials      0                                   \
#    --means         0.5466 0.6427 0.4240                \
#    --stds          0.1771 0.1648 0.2370
#
#time python ../classification/run.py                    \
#    --cuda_device   0                                   \
#    --save_model                                        \
#    --root_path     ../..                               \
#    --model_type    ResNet                              \
#    --test_hdf5     val_3.hdf5                          \
#    --train_hdf5    train_3.hdf5                        \
#    --pretrained                                        \
#    --weighted_loss                                     \
#    --cuda                                              \
#    --loading_epoch 0                                   \
#    --total_epochs  100                                 \
#    --outdim        2                                   \
#    --optim_type    AdamW                               \
#    --lr            1e-4                                \
#    --patience      25                                  \
#    --weight_decay  1e-4                                \
#    --scheduler                                         \
#    --bsize         128                                 \
#    --nworker       1                                   \
#    --cross_validation                                  \
#    --test_date    Jan_26_2026                          \
#    --study_name    downy_trial_11_24_24                \
#    --n_trials      0                                   \
#    --means         0.5466 0.6427 0.4240                \
#    --stds          0.1771 0.1648 0.2370
#
  time python ../classification/run.py                    \
      --cuda_device   0                                   \
      --save_model                                        \
      --root_path     ../..                               \
      --model_type    ResNet                              \
      --test_hdf5     val_4.hdf5                          \
      --train_hdf5    train_4.hdf5                        \
      --pretrained                                        \
      --weighted_loss                                     \
      --cuda                                              \
      --loading_epoch 0                                   \
      --total_epochs  100                                 \
      --outdim        2                                   \
      --optim_type    AdamW                               \
      --lr            1e-4                                \
      --patience      25                                  \
      --weight_decay  1e-4                                \
      --scheduler                                         \
      --bsize         128                                 \
      --nworker       1                                   \
      --cross_validation                                  \
      --test_date    Jan_26_2026                          \
      --study_name    downy_trial_11_24_24                \
      --n_trials      0                                   \
      --means         0.5466 0.6427 0.4240                \
      --stds          0.1771 0.1648 0.2370


echo "Training run completed."



# to view in tensorboard:
# python -m tensorboard.main --logdir "../../results/runs"