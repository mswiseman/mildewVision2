# Preface
The Blackbird is a highthroughput phenomics imaging platform developed through collaboration of scientists and engineers at [Cornell AgriTech](https://cals.cornell.edu/cornell-agritech), the [USDA-ARS Grape Genetics Research Unit (GGRU)](https://www.ars.usda.gov/northeast-area/geneva-ny/grape-genetics-research-unit-ggru/), and [Moblanc Robotics](https://moblancrobotics.com/). Most code/scripts in this repository build off of [Tian Qiu's Grape PM Saliency mapping repository](https://github.com/suptimq/Saliency_based_Grape_PM_Quantification) (used for [this paper](https://academic.oup.com/hr/article/doi/10.1093/hr/uhac187/6675613)). 


This codebase is a continual work-in-progress as I'm always trying to improve our code, documentation, and models; alas, feel free to email me with any questions or clarifications: [wisemami@oregonstate.edu](mailto:wisemami@oregonstate.edu) 

# Introduction

The code in this repository primarily uses [PyTorch](https://pytorch.org/get-started/locally/) pretrained models to train and subsequently make inferences on leaf disks with or without powdery mildew. <br>

Overview of the training and inference process: <br>
<p align="center">
  <img src="manuscript_figures/fig_2.png" width="1000">
</p>

# Implementation

[CUDA](https://developer.nvidia.com/cuda-toolkit) is required for GPU usage; currently it's only available for PCs. Please check your GPU to figure out which version you need. If running on Apple Silicon, [MPS](https://developer.apple.com/metal/pytorch/) is necessary to take advantage of accelerated PyTorch. <br>

**Package Requirements**: <br>
To install the required packages via conda, simply run `conda env create -f code/environment.yml` and then `conda activate mildewVision` to activate the environment.   <br><br>If running on **Google Colab**, check out a GPU (preferably A100 or better when training) and run: `!pip install optuna==3.1.0 termcolor` as the other packages should already be installed (as of 11/25/2025).  

![overview part 2](manuscript_figures/fig_3.png)

## Classification Training
To train your own model, you need:<br>

1. A labeled image patch dataset to build the necessary train/test/val .hdf5 files
   - you can make image patches using [code/common/makePatches.py](code/common/make_patches.py). It's easiest to sort these patches into different directories according to the label (e.g. if infected, put in the "infected" directory. If not infected, put in the "healthy" directory)
   - In subsequent models, I would make patches by using the `--save_infected`, `--save_healthy` or `--save_discarded` tags when running [plot_sal_map_leaf.py](code/plot_sal_map_leaf.py), this way I could correct and add previously missclassified patches to my new dataset in hopes the next model iteration would learn the features better. 
   - you can then make a train/test/val hdf5 files (or k-fold splits) using [code/common/images_to_test_train_hdf5.py](code/common/images_to_test_train_hdf5.py)
   - alternatively, you can reproduce our training by downloading the train/val/test splits used for our Jan26 model [here](10.5281/zenodo.19897533). 

2. To determine mean rgb chanel values for your test/train/val sets using [code/common/get_mean_std.py](code/common/get_mean_std.py) and plug those into your [code/script/train.sh](code/script/train.sh) script under `--means` and `--stds` (super important...this dramatically effects your model performance). 

3. Customize other training parameters such as the model, learning rate, etc. within the [code/script/train.sh](code/script/train.sh) script. See the argparse section in [code/classification/run.py](code/classification/run.py) to see full list of customizable variables. <br><br> Note: You can start with the default values, but your model will likely perform better if you try different base models and hyperparamter values (e.g. by using [Optuna](https://optuna.org/) hyperparameter optimization). Always cross-validate and test to ensure you're not overfitting though. 


## Inference
Once you have downloaded our example [two-class hop powdery mildew model]() to [results/ResNet_Jan26_23-15-35_2026](results/ResNet_Jan26_23-15-35_2026), you should be able to activate the conda environment (`conda activate mildewVision`) and run either `bash ./code/script/leaf_correlation_all.sh` or `bash ./code/script/plot_sal_map_leaf.sh` as a minimal working example. 

Once that's working, you can customize the argparse arguments in the [leaf_correleation_all.sh](code/script/leaf_correlation_all.sh) bash script to run inference on multiple datasets in parallel (adjust the `MAX_JOBS` parameter according to your computational power).  In the example [here](code/script/leaf_correlation_all.sh), I have included commands for calling either [plot_sal_map_leaf.py](code/plot_sal_map_leaf.py) or  [leaf_correlation_mw.py](code/leaf_correlation_mw.py). Both code/script return the same .csv file that provides metadata about your run parameters, disease severity estimates, saliency metrics, etc., but [plot_sal_map_leaf.py](code/plot_sal_map_leaf.py) also returns visual outputs of patch disease severity as well as saliency maps (if you include the optional saliency tags, see example below). If you are running standard inference you may opt to call the [code/leaf_correlation_mw.py](code/leaf_correlation_mw.py) [script](code/script/leaf_correlation_all.sh) instead as it runs 5-10x faster. 

<br>Example raw, patch and saliency maps output of from `plot_sal_map_leaf.py`:

<p align="center">
  <img src="manuscript_figures/fig_6.png" width="600">
</p>

## Segmentation Training
*Coming soon...*

## Testing
*Coming soon...*

## Image data
1 cm leaf disks were excised using ethanol disinfested leather punches and subsequently arrayed adaxial side onto up on 1% water agar plates. Image acquisition was performed using the Blackbird CNC Imaging Robot (version 1 "Blackbird-Green", developed by Cornell University, USDA-ARS Grape Genetics Research Unit, and Moblanc Robotics).  The Blackbird is a G-code driven CNC that positions a Nikon Z 7II mirrorless camera equipped with a 2.5x zoom ultra-macro lens (Venus Optics Laowa 25mm) in the X/Y position and then the camera captures images in a z-stack every 200 µM in Z-height. The image stacking process is automated using the [stackPhotosParallel.py](code/common/stackPhotosParallel.py). [Helicon Focus software](https://www.heliconsoft.com/software-downloads/) (Helicon Software, version 8.1) was utilized to perform the focus stacking, with the parameters set to method B (depth map radius: 1, smoothing radius: 4, and sharpness: 2). <br><br> The test set used for assessing concordance with human raters can be downloaded [here](10.5281/zenodo.19897533).
![blackbird robot](manuscript_figures/fig_1.png)

## File structure of repository

```
miteVision2/
├── code
│   ├── analysis
│   ├── classification
│   ├── segmentation
│   ├── visualization
│   ├── metric
│   ├── common
│   ├── script
│   │   ├── inference.sh
│   │   ├── inference_seg.sh
│   │   ├── leaf_correlation_all.sh
│   │   ├── plot_leaf_correlation_all.sh
│   │   ├── plot_sal_map_leaf.sh
│   │   ├── plot_sal_map_patch.sh
│   │   ├── train.sh
│   │   └── train_seg.sh
│   ├── figures
│   ├── sanity_check
│   ├── analyzer_config.py
│   ├── utils.py
│   ├── environment.yml
│   ├── README.md
│   ├── leaf_correlation_mw.py
│   ├── plot_sal_map_leaf.py
│   ├── plot_sal_map_leaf_fixed_optimized_th.py
│   ├── plot_sal_map_patch.py
│   └── plot_sal_map_patch_seg_iou.py
│
├── data
│   ├── patches_for_class_visualization
│   ├── human_disk_assessments/  # need to download from zenodo
│   │   ├── 1-22-2026_5dpi/1
│   │   ├── 1-22-2026_10dpi/1
│   └── 6-28-2023_10dpi/1
│
├── results
│   ├── models/ResNet_Jan26_23-15-35_2026 # need to download from zenodo 
│   ├── logs
│   │   ├── asabe_journal
│   │   ├── cls_cv
│   │   ├── seg
│   │   └── grid_logs
│   ├── runs
│   │   ├── asabe_journal
│   │   ├── cls_cv
│   │   └── seg
│   ├── segmentation
│   │   ├── 6-28-2023_10dpi-1
│   │   ├── human_disk_assessments
│   │   └── time_series
│   └── journal
│       └── inference_results
│
├── r
│   ├── figures
│   │   ├── agreement_plots
│   │   ├── correlation_plots
│   │   ├── training_curves
│   │   └── validation_figures
│   ├── cv_training_plots.Rmd
│   ├── model_concordance_with_humans.Rmd
│   └── gh_v_cv_models_mapping_population_streamlined.Rmd
│
├── spreadsheets
    ├── human_patch_ratings
        ├── *_response.csv
        └── human_vs_blackbird_patches.xlsx
    ├── 2023 Multiparent PM Phenotyping project Datasheet.xlsx
    ├── best_model_training_stats_jan_26_26_ep_44.csv
    ├── segmentation_results.csv
    ├── human_vs_bb_models_disase_severity.csv
    ├── human_vs_bb_models_disase_severity.csv
    └── mapping_population_Jan26_26_ResNet_model_data_*.xlsx
```




