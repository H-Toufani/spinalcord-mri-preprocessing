# Boundary-Aware Preprocessing and Deep Learning Segmentation for Spinal Cord MRI

A multi-stage MRI preprocessing framework followed by 3D U-Net segmentation using MONAI and PyTorch, designed for spinal cord MRI acquired across institutions under heterogeneous imaging conditions.

## Overview

- **Preprocessing:** image-quality assessment, spatial standardization, intensity correction and normalization, contrast enhancement, edge-aware filtering, and quantitative evaluation.
- **Deep learning:** 3D U-Net training with single-channel and multi-channel inputs, data augmentation, cross-dataset evaluation, and performance analysis.

## Repository structure

```text
src/
├── preprocessing/
│   ├── 00_dataset_scan_stats.py
│   ├── 01_resample.py
│   ├── 02_crop.py
│   ├── 03_n4_bias_correction.py
│   ├── 04_nlm_denoising.py
│   ├── 05_zscore_normalization.py
│   ├── 06_clahe_histogram_equalization.py
│   └── 07_log_filter.py
└── training/
    ├── 01_train_unet_no_augmentation.py
    ├── 02_train_unet_with_augmentation.py
    └── 03_train_unet_3channel.py
requirements.txt
LICENSE
README.md
```

## Data

The study uses three institutional datasets, referred to as Site1, Site2, and Site3. Raw MRI data are not publicly distributed because of patient privacy and institutional restrictions.

Expected subject-folder structure:

```text
DatasetRoot/
├── Subject_001/
│   ├── T2s.nii.gz
│   └── T2s_manual_seg.nii.gz
└── Subject_002/
    └── ...
```

Each subject folder must contain the image and its corresponding segmentation mask. Configure input and output paths for your local datasets before running the scripts.

## Installation

### 1. Clone the repository

```bash
git clone https://github.com/H-Toufani/spinalcord-mri-preprocessing.git
cd spinalcord-mri-preprocessing
```

### 2. Create an environment

Using Conda:

```bash
conda create -n sc_mri python=3.9
conda activate sc_mri
```

Alternatively, using venv on macOS or Linux:

```bash
python -m venv venv
source venv/bin/activate
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

Main libraries include PyTorch, MONAI, NumPy, SciPy, NiBabel, SimpleITK, DIPY, scikit-image, OpenPyXL, and Matplotlib. See `requirements.txt` for the dependency list.

## Preprocessing workflow

Scripts are located in `src/preprocessing/` and follow this sequence:

| Step | Script | Purpose |
| --- | --- | --- |
| 0 | `00_dataset_scan_stats.py` | Extract image resolution, spacing, and volume statistics |
| 1 | `01_resample.py` | Resample volumes to unified voxel spacing |
| 2 | `02_crop.py` | Center crop and pad to 160 × 160 × 16 |
| 3 | `03_n4_bias_correction.py` | N4 bias-field correction and coefficient-of-variation assessment |
| 4 | `04_nlm_denoising.py` | Non-local means denoising and SNR/CNR assessment |
| 5 | `05_zscore_normalization.py` | Normalize intensities within the body region |
| 6 | `06_clahe_histogram_equalization.py` | Slice-wise contrast enhancement |
| 7 | `07_log_filter.py` | Laplacian-of-Gaussian edge representations |

Run each script with the appropriate local paths and settings. For example, from the preprocessing directory:

```bash
cd src/preprocessing
python 00_dataset_scan_stats.py
```

The workflow includes quantitative reports for statistical analysis.

## Deep learning training

Training scripts are located in `src/training/`:

- `01_train_unet_no_augmentation.py`: baseline U-Net without augmentation.
- `02_train_unet_with_augmentation.py`: U-Net with augmentation.
- `03_train_unet_3channel.py`: multi-channel U-Net using raw, CLAHE-enhanced, and LoG-filtered images.

Example from the training directory:

```bash
python 03_train_unet_3channel.py \
  --Site1_root /path/to/Site1 \
  --Site2_root /path/to/Site2 \
  --Site3_root /path/to/Site3
```

Replace the example dataset paths with your own.

## Evaluation and reproducibility

The training workflows evaluate Dice similarity coefficient, precision, and recall on held-out test sets, and save training curves and checkpoints.

For comparable experiments, preserve the preprocessing order, random seed, dataset splits, and model settings. Record the environment and any changes to default parameters alongside the results.

## Ethics and data availability

MRI datasets were anonymized and used under institutional ethics approvals and data-sharing agreements. Data access is subject to institutional restrictions and ethical approval; the clinical datasets are not included here. Contact the corresponding researcher for data-access inquiries.

## License

Code is available under the [MIT License](LICENSE).

## Contact

**Hediyeh Toufani**  
PhD Candidate, Biomedical Engineering · University of Ottawa

[Email](mailto:h.toufani@uottawa.ca) · [LinkedIn](https://www.linkedin.com/in/hediyehtoufani/) · [Google Scholar](https://scholar.google.com/citations?user=D49VhZYAAAAJ&hl=en)
