#!/bin/bash
#
#
#python ../segmentation/infer_from_folder.py                                                           \
#    --model_type        DeepLab                                                                       \
#    --loading_epoch     60                                                                            \
#    --pretrained                                                                                      \
#    --timestamp         Feb04_00-32-07_2021                                                           \
#    --model_path        "../.."                               \
#    --cuda                                                                                            \
#    --cuda_id           0                                                                             \
#    --patch_folder      "../../data/segmentation/test_set/images" \
#    --out_mask_folder   "../../data/segmentation/test_set/masks"
#
#

#
#    python ../segmentation/infer_from_folder.py                                                           \
#        --model_type        DeepLab                                                                       \
#        --loading_epoch     60                                                                            \
#        --pretrained                                                                                      \
#        --timestamp         Feb04_00-32-07_2026                                                           \
#        --model_path        "../.."                                                                       \
#        --cuda                                                                                            \
#        --cuda_id           0                                                                             \
#        --patch_folder      "../../../../Downloads/segmentation/test/images"                                     \
#        --out_mask_folder   "../../../../Downloads/segmentation/test/masks"


    python ../segmentation/infer_full_leaf_seg.py                                                         \
        --model_type        DeepLab                                                                       \
        --loading_epoch     60                                                                            \
        --pretrained                                                                                      \
        --timestamp         Feb04_00-32-07_2026                                                           \
        --model_path        "../.."                                                                       \
        --cuda                                                                                            \
        --cuda_id           0                                                                             \
        --outdim            2                                                                             \
        --in_folder         "../../data/human_disk_assessments/1-22-26/10dpi"                                                \
        --out_folder        "../../results/human_disk_assessments/segmentation/DeepLab-Feb04_00-32-07_2026"                                \
        --step              112                                                                           \
        --batch_size        8
