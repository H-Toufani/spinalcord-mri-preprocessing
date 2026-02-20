"""
Summary:
    Resamples all T2s.nii.gz volumes under a dataset root directory
    to a fixed target voxel spacing using affine-aware resampling.

    Linear interpolation (order=1) is used for intensity images.
    If you resample masks, you must use order=0 (nearest neighbor).

Inputs:
    - A root directory containing subject subfolders with T2s.nii.gz files.

Outputs:
    - A resampled NIfTI file saved in the same folder as the input,
      with a configurable suffix (default: T2s_resampled.nii.gz).

How to run:
    pip install nibabel numpy
    python 01_resample.py --root_dir /path/to/dataset/root
"""

import os
import argparse
import nibabel as nib
import numpy as np
from nibabel.processing import resample_to_output


def parse_args():
    parser = argparse.ArgumentParser(description="Resample T2s volumes to fixed voxel spacing.")
    parser.add_argument(
        "--root_dir",
        type=str,
        required=False,
        default=os.environ.get("DATA_ROOT_DIR", ""),
        help="Path to dataset root directory (or set DATA_ROOT_DIR environment variable).",
    )
    parser.add_argument(
        "--filename",
        type=str,
        default="T2s.nii.gz",
        help='Input filename to search for (default: "T2s.nii.gz").',
    )
    parser.add_argument(
        "--out_suffix",
        type=str,
        default="T2s_resampled.nii.gz",
        help='Output filename (default: "T2s_resampled.nii.gz").',
    )
    parser.add_argument(
        "--spacing",
        type=float,
        nargs=3,
        default=(0.625, 0.625, 4.0),
        help="Target voxel spacing in mm (sx sy sz). Default: 0.625 0.625 4.0",
    )
    return parser.parse_args()


def main():
    args = parse_args()

    root_dir = args.root_dir
    filename = args.filename
    out_suffix = args.out_suffix
    target_spacing = tuple(args.spacing)

    if not root_dir or not os.path.isdir(root_dir):
        raise ValueError("Invalid root_dir. Provide --root_dir or set DATA_ROOT_DIR.")

    count = 0

    for root, dirs, files in os.walk(root_dir):
        if filename in files:
            in_path = os.path.join(root, filename)

            print("Reading:", in_path)

            img = nib.load(in_path)

            print("  Original shape  :", img.shape)
            print("  Original spacing:", img.header.get_zooms()[:3])

            # affine-aware resampling
            img_resampled = resample_to_output(
                img,
                voxel_sizes=target_spacing,
                order=1   # linear interpolation for intensity images
            )

            print("  Resampled shape :", img_resampled.shape)
            print("  Resampled spacing:", img_resampled.header.get_zooms()[:3])

            out_path = os.path.join(root, out_suffix)
            nib.save(img_resampled, out_path)

            print("  Saved:", out_path)
            print("-" * 50)

            count += 1

    print("Total resampled volumes:", count)


if __name__ == "__main__":
    main()
