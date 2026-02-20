# Boundary-Aware Preprocessing and Deep Learning Segmentation for Spinal Cord MRI

This repository contains the full preprocessing and deep learning pipeline used in the study:

**“Boundary-Aware Preprocessing for Robust Spinal Cord MRI Segmentation under Deformation and Compression”**

The project implements a multi-stage MRI preprocessing framework followed by 3D U-Net–based segmentation using MONAI and PyTorch. It is designed for spinal cord MRI data acquired from multiple institutions under heterogeneous imaging conditions.

---

## Overview

The pipeline consists of two main components:

1. Preprocessing Module  
   - Image quality assessment  
   - Spatial standardization  
   - Intensity correction and normalization  
   - Contrast enhancement  
   - Edge-aware filtering  
   - Quantitative evaluation  

2. Deep Learning Module  
   - 3D U-Net training  
   - Single-channel and multi-channel inputs  
   - Data augmentation  
   - Cross-dataset evaluation  
   - Performance analysis  

All scripts are organized sequentially and can be executed independently.

---

## Repository Structure

├── src/
│ ├── preprocessing/
│ │ ├── 00_dataset_scan_stats.py
│ │ ├── 01_resample.py
│ │ ├── 02_crop.py
│ │ ├── 03_n4_bias_correction.py
│ │ ├── 04_nlm_denoising.py
│ │ ├── 05_zscore_normalization.py
│ │ ├── 06_clahe_histogram_equalization.py
│ │ └── 07_log_filter.py
│ │
│ └── training/
│ ├── 01_train_unet_no_augmentation.py
│ ├── 02_train_unet_with_augmentation.py
│ ├── 03_train_unet_3channel.py
│
├── requirements.txt
├── LICENSE
└── README.md


---

## Datasets

This study uses three institutional datasets:

- Site1  
- Site2  
- Site3  

Due to privacy and ethical restrictions, raw MRI data are not publicly available.

All scripts assume the following folder structure:

DatasetRoot/
├── Subject_001/
│ ├── T2s.nii.gz
│ ├── T2s_manual_seg.nii.gz
│ └── ...
├── Subject_002/
│ └── ...


Each subject folder must contain the corresponding image and segmentation mask.

---

## Installation

### 1. Clone the repository

```bash
git clone https://github.com/yourusername/your-repo-name.git
cd your-repo-name
2. Create environment (recommended)
Using Conda:

conda create -n sc_mri python=3.9
conda activate sc_mri
Or using venv:

python -m venv venv
source venv/bin/activate
3. Install dependencies
pip install -r requirements.txt
Requirements
Main libraries:

Python ≥ 3.8

PyTorch

MONAI

NumPy

SciPy

NiBabel

SimpleITK

DIPY

Scikit-image

OpenPyXL

Matplotlib

A complete list is provided in requirements.txt.

--------------

Preprocessing Pipeline
All preprocessing scripts are located in:

src/preprocessing/
They should be executed in order.

Step 0: Dataset Statistics
python 00_dataset_scan_stats.py
Extracts image resolution, spacing, and volume statistics.

Step 1: Resampling
python 01_resample.py
Resamples all volumes to a unified voxel spacing.

Step 2: Cropping
python 02_crop.py
Performs center cropping and padding to fixed size (160×160×16).

Step 3: N4 Bias-Field Correction
python 03_n4_bias_correction.py
Applies N4 correction and evaluates coefficient of variation.

Step 4: Non-Local Means Denoising
python 04_nlm_denoising.py
Performs noise suppression and computes SNR/CNR metrics.

Step 5: Z-Score Normalization
python 05_zscore_normalization.py
Normalizes intensities within body region.

Step 6: CLAHE Enhancement
python 06_clahe_histogram_equalization.py
Applies slice-wise contrast-limited adaptive histogram equalization.

Step 7: Laplacian-of-Gaussian Filtering
python 07_log_filter.py
Extracts edge-aware representations.

All steps generate quantitative Excel reports for statistical analysis.

------------------------

Deep Learning Training
Training scripts are located in:

src/training/
Baseline U-Net (No Augmentation)
python 01_train_unet_no_augmentation.py

U-Net with Data Augmentation
python 02_train_unet_with_augmentation.py

Multi-Channel U-Net (3 Inputs)
python 03_train_unet_3channel.py \
  --Site1_root /path/to/Site1 \
  --Site2_root /path/to/Site2 \
  --Site3_root /path/to/Site3
Uses Raw, CLAHE-enhanced, and LoG-filtered images as three complementary channels.

------------

Evaluation
All training scripts compute:

Dice Similarity Coefficient

Precision

Recall

on held-out test sets.

Training curves and checkpoints are automatically saved.

Reproducibility
To reproduce experiments:

Use the same preprocessing order

Fix random seed (default = 42)

Use identical data splits

Run training scripts with default parameters

All scripts explicitly set random seeds.

Ethics Statement
This study was conducted in accordance with institutional ethical regulations.

All MRI datasets were anonymized prior to analysis and were used under approved data-sharing agreements.

No personally identifiable information is contained in this repository.

Data Availability
Due to patient privacy and institutional restrictions, the datasets used in this study are not publicly available.

Data access may be granted upon reasonable request to the corresponding author, subject to ethical approval.

Code Availability
All preprocessing and training codes are available in this repository.

The code is provided for academic and research purposes.

License
This project is licensed under the MIT License. See LICENSE for details.


Contact
Hediyeh Toufani
PhD Candidate, Biomedical Engineering
University of Ottawa
h.toufani@uottawa.ca

For questions or collaboration inquiries, please contact via GitHub or institutional email.

