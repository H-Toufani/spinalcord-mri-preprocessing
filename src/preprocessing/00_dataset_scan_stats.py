"""
Summary:
    Scans a dataset folder for volumes named "T2s.nii.gz", extracts each image's
    spatial shape (X, Y, Z) and voxel spacing (sx, sy, sz), and exports:
      1) an Excel file (per-case table + summary stats)

Inputs:
    - A root directory that contains subject subfolders (any depth), where each
      subject folder includes a file named "T2s.nii.gz".

Outputs (saved in the current working directory by default):
    - t2s_scan_table.xlsx

How to run:
    1) Install dependencies:
        pip install numpy nibabel openpyxl
    2) Run:
        python 00_dataset_scan_summary.py --root_dir /path/to/dataset/root

"""

import os
import argparse
import numpy as np
import nibabel as nib

from openpyxl import Workbook


def parse_args():
    parser = argparse.ArgumentParser(description="Generate shape/spacing summary for T2s.nii.gz volumes.")
    parser.add_argument(
        "--root_dir",
        type=str,
        required=False,
        default=os.environ.get("DATA_ROOT_DIR", ""),
        help=(
            "Path to dataset root directory. "
            "You can also set the environment variable DATA_ROOT_DIR instead."
        ),
    )
    parser.add_argument(
        "--filename",
        type=str,
        required=False,
        default="T2s.nii.gz",
        help='Name of the NIfTI file to look for (default: "T2s.nii.gz").',
    )
    parser.add_argument(
        "--excel_out",
        type=str,
        required=False,
        default="t2s_scan_table.xlsx",
        help="Output Excel filename (default: t2s_scan_table.xlsx).",
    )
    
    return parser.parse_args()


def main():
    args = parse_args()

    root_dir = args.root_dir
    target_filename = args.filename

    # ---- basic checks (kept minimal; does not change processing logic) ----
    if not root_dir or not isinstance(root_dir, str):
        raise ValueError(
            "root_dir is empty. Provide --root_dir /path/to/data or set DATA_ROOT_DIR environment variable."
        )
    if not os.path.isdir(root_dir):
        raise ValueError(f"root_dir does not exist or is not a directory: {root_dir}")

    rows = []          # for table
    shapes = []        # list of (X,Y,Z)
    spacings = []      # list of (sx,sy,sz)

    for root, dirs, files in os.walk(root_dir):
        if target_filename in files:
            file_path = os.path.join(root, target_filename)

            img = nib.load(file_path)

            shape = img.shape[:3]
            spacing = img.header.get_zooms()[:3]

            folder_name = os.path.basename(root)

            rows.append([
                folder_name,
                file_path,
                int(shape[0]), int(shape[1]), int(shape[2]),
                float(spacing[0]), float(spacing[1]), float(spacing[2])
            ])

            shapes.append(shape)
            spacings.append(spacing)

    # basic check
    if len(rows) == 0:
        raise ValueError(f'No "{target_filename}" files found under: {root_dir}')

    shapes_arr = np.array(shapes, dtype=np.float64)      # N x 3
    spacings_arr = np.array(spacings, dtype=np.float64)  # N x 3

    min_shape = np.min(shapes_arr, axis=0)        # per-dimension min
    median_shape = np.median(shapes_arr, axis=0)  # per-dimension median

    min_spacing = np.min(spacings_arr, axis=0)        # per-dimension min
    median_spacing = np.median(spacings_arr, axis=0)  # per-dimension median

    min_z_slices = int(min_shape[2])
    median_z_slices = float(median_shape[2])

    min_z_spacing = float(min_spacing[2])
    median_z_spacing = float(median_spacing[2])

    print("Total volumes:", len(rows))
    print("Smallest shape (X,Y,Z):", tuple(min_shape.astype(int)))
    print("Median shape (X,Y,Z):", (float(median_shape[0]), float(median_shape[1]), float(median_shape[2])))
    print("Smallest spacing (sx,sy,sz) mm:", tuple(min_spacing))
    print("Median spacing (sx,sy,sz) mm:", tuple(median_spacing))
    print("Z only: smallest Z_slices =", min_z_slices, ", median Z_slices =", median_z_slices)
    print("Z only: smallest sz(mm) =", min_z_spacing, ", median sz(mm) =", median_z_spacing)

    # ----------------------------
    # Write Excel
    # ----------------------------
    wb = Workbook()
    ws = wb.active
    ws.title = "T2s Scan Summary"

    headers = ["Folder", "FilePath", "X", "Y", "Z", "sx(mm)", "sy(mm)", "sz(mm)"]
    ws.append(headers)

    for r in rows:
        ws.append(r)

    # Summary sheet
    ws2 = wb.create_sheet("Summary")
    ws2.append(["Metric", "X", "Y", "Z"])
    ws2.append(["Min shape", int(min_shape[0]), int(min_shape[1]), int(min_shape[2])])
    ws2.append(["Median shape", float(median_shape[0]), float(median_shape[1]), float(median_shape[2])])

    ws2.append(["", "", "", ""])
    ws2.append(["Metric", "sx(mm)", "sy(mm)", "sz(mm)"])
    ws2.append(["Min spacing", float(min_spacing[0]), float(min_spacing[1]), float(min_spacing[2])])
    ws2.append(["Median spacing", float(median_spacing[0]), float(median_spacing[1]), float(median_spacing[2])])

    excel_out = args.excel_out
    wb.save(excel_out)
    print("Saved Excel:", excel_out)

    # ----------------------------

if __name__ == "__main__":
    main()
