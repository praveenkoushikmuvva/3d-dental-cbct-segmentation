# 3d-dental-cbct-segmentation
# DobbeAI: 3D Dental Segmentation

![Python](https://img.shields.io/badge/Python-3.8%2B-blue)
![PyTorch](https://img.shields.io/badge/PyTorch-EE4C2C?logo=pytorch&logoColor=white)

This repository contains our end-to-end deep learning pipeline for 3D instance segmentation of dental CBCT scans. The model automatically identifies and segments the jaw and up to 42 individual teeth, outputting the results into a lightweight, interactive web viewer.

![Viewer Demo](assets/demo.gif)  
*(Note: Add your GIF here showing the viewer in action)*

## Project Overview

We built this pipeline to handle the complexities of real-world dental scans, specifically focusing on handling metal artifacts, variable scan resolutions, and instance-level tooth identification. 

* **Dataset:** Trained on the public **ToothFairy2** dataset. We used a strict, stratified 70/15/15 split for training, validation, and testing.
* **Architecture:** 3D U-Net implemented via MONAI and PyTorch.
* **Viewer:** A zero-dependency, local web viewer built with NiiVue. It runs entirely in the browser via WebGL.

## How It Works

### 1. Data Preparation
Raw `.mha` files from the dataset are processed into the standard `nnU-Net` JSON structure. We re-mapped the standard FDI tooth numbering (which has numerical gaps) into 43 consecutive classes (0 for background, 1-42 for individual teeth) to stabilize the loss function during training.

### 2. Training Pipeline
We used MONAI to handle the heavy lifting for 3D volumetric data. Key preprocessing steps include:
* **Isotropic Resampling:** All scans are forced to `0.5 x 0.5 x 0.5 mm` voxel spacing.
* **Intensity Windowing:** Hounsfield Units (HU) are clipped between `0` and `1500`. This ignores soft tissue/air and forces the model to focus strictly on dentin and enamel contrast.
* **Augmentations:** Random spatial cropping (128³ patches), axis flipping, rotations, and Gaussian noise injections to prevent overfitting.

### 3. Inference & Visualization
The inference script uses sliding-window evaluation to reconstruct the full 512x512xZ volume without running out of GPU memory. The output is saved as a NIfTI file (`mask.nii.gz`) with the affine matrix preserved, ensuring perfect alignment when overlaid on the original scan.

## Repository Structure

```text
dobbe-3d-dental/
├── data_prep/
│   ├── nnUNet_ToothFairy2_Preprocessing.ipynb  # Initial data cleaning & JSON setup
│   └── dataset.json                            
├── model_weights/
│   └── dental_unet_final.pth                   # Pre-trained weights (19MB)
├── model_training/
│   └── Monai_3D_UNet_Training.ipynb            # Core training and inference loop
├── web_viewer/
│   └── index.html                              # Local 3D viewer
├── demo_data/
│   ├── scan.nii.gz                             # Sample CBCT
│   └── mask.nii.gz                             # AI output
├── requirements.txt                            
└── README.md
