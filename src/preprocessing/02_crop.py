"""
Summary:
    Center-crops (or zero-pads) a 3D segmentation volume to a fixed size:
        - X,Y: target_xy x target_xy (default: 160 x 160)
        - Z  : target_z slices (default: 16)

    The crop/pad is applied around the volume center.
    The NIfTI affine is updated to preserve correct world-space coordinates.

Typical use:
    Standardize image dimensions before training/evaluation.

Inputs:
    - A root directory containing subject subfolders with segmentation files
      (default input name: "T2s_resampled.nii.gz").

Outputs:
    - A cropped/padded segmentation saved alongside the input
      (default output name: "T2s_cropped.nii.gz").

How to run:
    pip install nibabel numpy
    python 02_crop_pad_segmentation.py --root_dir /path/to/dataset/root
"""

import os
import argparse
import numpy as np
import nibabel as nib


def parse_args():
    parser = argparse.ArgumentParser(description="Center crop/pad segmentation volumes to fixed shape.")
    parser.add_argument(
        "--root_dir",
        type=str,
        required=False,
        default=os.environ.get("DATA_ROOT_DIR", ""),
        help="Path to dataset root directory (or set DATA_ROOT_DIR environment variable).",
    )
    parser.add_argument(
        "--in_name",
        type=str,
        default="T2s_resampled.nii.gz",
        help='Input segmentation filename (default: "T2s_resampled.nii.gz").',
    )
    parser.add_argument(
        "--out_name",
        type=str,
        default="T2s_cropped.nii.gz",
        help='Output filename (default: "T2s_cropped.nii.gz").',
    )
    parser.add_argument(
        "--target_xy",
        type=int,
        default=160,
        help="Target size for X and Y dimensions (default: 160).",
    )
    parser.add_argument(
        "--target_z",
        type=int,
        default=16,
        help="Target number of slices in Z dimension (default: 16).",
    )
    return parser.parse_args()


def main():
    args = parse_args()

    root_dir = args.root_dir
    in_name = args.in_name
    out_name = args.out_name
    target_xy = args.target_xy
    target_z = args.target_z

    if not root_dir or not os.path.isdir(root_dir):
        raise ValueError("Invalid root_dir. Provide --root_dir or set DATA_ROOT_DIR.")

    count = 0

    for root, dirs, files in os.walk(root_dir):
        if in_name in files:
            in_path = os.path.join(root, in_name)
            print("Processing:", in_path)

            img = nib.load(in_path)
            vol = img.get_fdata().astype(np.float32)
            affine = img.affine.copy()

            # ---------------------------
            # Center Z crop/pad (target_z slices)
            # ---------------------------
            z = vol.shape[2]

            if z >= target_z:
                start_z = (z - target_z) // 2
                vol = vol[:, :, start_z:start_z + target_z]

                # update affine origin in Z
                affine[:3, 3] += affine[:3, 2] * start_z
            else:
                pad_before = (target_z - z) // 2
                pad_after = target_z - z - pad_before

                vol = np.pad(vol, ((0, 0), (0, 0), (pad_before, pad_after)), mode="constant")

                affine[:3, 3] -= affine[:3, 2] * pad_before

            # ---------------------------
            # Center X,Y crop/pad (target_xy x target_xy)
            # ---------------------------
            x, y = vol.shape[0], vol.shape[1]

            pad_x_before = max(0, (target_xy - x) // 2)
            pad_x_after  = max(0, target_xy - x - pad_x_before)
            pad_y_before = max(0, (target_xy - y) // 2)
            pad_y_after  = max(0, target_xy - y - pad_y_before)

            if pad_x_before or pad_x_after or pad_y_before or pad_y_after:
                vol = np.pad(
                    vol,
                    ((pad_x_before, pad_x_after),
                     (pad_y_before, pad_y_after),
                     (0, 0)),
                    mode="constant"
                )

                affine[:3, 3] -= (
                    affine[:3, 0] * pad_x_before +
                    affine[:3, 1] * pad_y_before
                )

            x, y = vol.shape[0], vol.shape[1]
            start_x = (x - target_xy) // 2
            start_y = (y - target_xy) // 2

            vol = vol[start_x:start_x + target_xy,
                      start_y:start_y + target_xy,
                      :]

            affine[:3, 3] += (
                affine[:3, 0] * start_x +
                affine[:3, 1] * start_y
            )

            # ---------------------------
            # Final sanity check
            # ---------------------------
            if vol.shape != (target_xy, target_xy, target_z):
                raise ValueError(f"Wrong final shape {vol.shape} for {in_path}")

            # ---------------------------
            # Save output
            # ---------------------------
            out_path = os.path.join(root, out_name)
            out_img = nib.Nifti1Image(vol, affine)
            nib.save(out_img, out_path)

            print("Saved:", out_path, "Shape:", vol.shape)
            print("-" * 50)

            count += 1

    print("Total saved volumes:", count)


if __name__ == "__main__":
    main()
