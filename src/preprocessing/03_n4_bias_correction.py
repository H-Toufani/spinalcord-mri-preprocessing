"""
Summary:
    N4 bias-field correction + ROI CoV evaluation (per-image, per-group, overall) + Excel export.

What this script does:
    1) For each dataset group (multiple roots), finds subject folders containing:
         - an input image (default: T2s_cropped.nii.gz)
         - an ROI mask (default: T2s_seg_cropped.nii.gz)
    2) Runs 3D N4 bias-field correction (SimpleITK) and saves the corrected image per subject folder.
    3) Uses the ROI mask to compute cord statistics before/after correction:
         mean (mu), standard deviation (sigma), CoV = sigma / mu
    4) Exports results to an Excel workbook:
         - PerImage sheet: per-subject rows
         - GroupSummary sheet: per-group means + overall means across all groups

Requirements:
    pip install SimpleITK nibabel numpy openpyxl

Inputs (per subject folder):
    - Image: T2s_cropped.nii.gz   (configurable via --image_name)
    - Mask : T2s_seg_cropped.nii.gz (configurable via --mask_name)
      Note: Image and mask must already be aligned (same shape/orientation/space).

Outputs:
    - Corrected image saved in each subject folder (default name: T2s_N4_bias_field_correction.nii.gz)
    - Excel results file (default: 03_N4_CoV_results.xlsx)

How to run:
    python 03_N4_bias_correction.py \
        --Site1_root /path/to/Site1_dataset \
        --Site2_ /path/to/Site2_dataset \
        --Site3_root  /path/to/Site3_dataset
"""

import os
import argparse
import numpy as np
import nibabel as nib
import SimpleITK as sitk
from openpyxl import Workbook


def parse_args():
    parser = argparse.ArgumentParser(description="N4 bias correction + ROI CoV evaluation + Excel export.")

    # Dataset roots (set via args or environment variables)
    parser.add_argument("--Site1_root", type=str, default=os.environ.get("SITE1_ROOT", ""), help="Root directory for Site1.")
    parser.add_argument("--Site2_root", type=str, default=os.environ.get("SITE2_ROOT", ""), help="Root directory for Site2.")
    parser.add_argument("--Site3_root",  type=str, default=os.environ.get("SITE3_ROOT",  ""), help="Root directory for Site3.")

    parser.add_argument("--image_name", type=str, default="T2s_cropped.nii.gz", help='Input image filename in each subject folder.')
    parser.add_argument("--mask_name",  type=str, default="T2s_seg_cropped.nii.gz", help='ROI mask filename in each subject folder.')
    parser.add_argument("--out_image_name", type=str, default="T2s_N4_bias_field_correction.nii.gz", help="Output corrected image filename per subject.")

    parser.add_argument("--excel_out", type=str, default="03_N4_CoV_results.xlsx", help="Output Excel filename.")

    # N4 parameters (kept as your defaults)
    parser.add_argument("--n4_shrink_factor", type=int, default=2, help="Shrink factor for speed (default: 2).")
    parser.add_argument(
        "--n4_num_iterations",
        type=int,
        nargs=4,
        default=[50, 50, 30, 20],
        help="Multi-resolution iteration schedule (default: 50 50 30 20).",
    )
    parser.add_argument("--n4_convergence_thresh", type=float, default=1e-7, help="N4 convergence threshold (default: 1e-7).")

    return parser.parse_args()


# ------------------------
# Helper: compute ROI stats from numpy arrays
# ------------------------
def roi_stats(image_3d, mask_3d):
    vals = image_3d[mask_3d > 0]
    if vals.size == 0:
        return np.nan, np.nan, np.nan
    mu = float(np.mean(vals))
    sigma = float(np.std(vals, ddof=0))
    cov = float(sigma / mu) if mu != 0 else np.nan
    return mu, sigma, cov


def main():
    args = parse_args()
    groups = [
        {"name": "Site1",  "root": args.Site1_root},
        {"name": "Site2",  "root": args.Site2_root},
        {"name": "Site3", "root": args.Site3_root},
    ]

    image_name = args.image_name
    mask_name = args.mask_name
    out_image_name = args.out_image_name
    excel_out = args.excel_out

    n4_shrink_factor = args.n4_shrink_factor
    n4_num_iterations = args.n4_num_iterations
    n4_convergence_thresh = args.n4_convergence_thresh

    # Validate roots (skip empty ones; allows running on just 1 dataset if desired)
    valid_groups = []
    for g in groups:
        if g["root"] and os.path.isdir(g["root"]):
            valid_groups.append(g)
        else:
            if g["root"] != "":
                print(f'Warning: root not found or invalid for group "{g["name"]}": {g["root"]}')

    if len(valid_groups) == 0:
        raise ValueError(
            "No valid dataset roots provided. Use --Site1_root/--Site2_root/--Site3_root or set env vars "
            "SITE1_ROOT, SITE2_ROOT, SITE3_ROOT."
        )

    # ------------------------
    # Storage for results
    # ------------------------
    rows = []            # per-subject rows
    group_summaries = [] # per-group mean rows
    all_before = []      # list of (mu, sigma, cov) for before
    all_after = []       # list of (mu, sigma, cov) for after

    # ------------------------
    # Main loop over groups and subjects
    # ------------------------
    for g in valid_groups:
        gname = g["name"]
        groot = g["root"]

        print("\nProcessing group:", gname)
        print("Root:", groot)

        group_before = []
        group_after = []

        # walk all folders; treat a folder as a "subject" if it contains image_name and mask_name
        for root, dirs, files in os.walk(groot):
            if image_name in files and mask_name in files:
                folder_name = os.path.basename(root)
                img_path = os.path.join(root, image_name)
                msk_path = os.path.join(root, mask_name)
                out_path = os.path.join(root, out_image_name)

                # read image + mask with nibabel for ROI stats "before"
                img_nii = nib.load(img_path)
                msk_nii = nib.load(msk_path)

                img_np = img_nii.get_fdata().astype(np.float32)
                msk_np = msk_nii.get_fdata()

                # basic checks
                if img_np.shape[:3] != msk_np.shape[:3]:
                    print("Shape mismatch, skipping:", folder_name, "img:", img_np.shape, "mask:", msk_np.shape)
                    continue

                # binarize mask
                msk_bin = (msk_np > 0).astype(np.uint8)

                mu_b, sd_b, cov_b = roi_stats(img_np, msk_bin)

                # N4 bias correction with SimpleITK
                # Read with SimpleITK to preserve spacing/origin/direction properly
                sitk_img = sitk.ReadImage(img_path, sitk.sitkFloat32)
                sitk_msk = sitk.ReadImage(msk_path, sitk.sitkUInt8)

                # Ensure mask is binary in SITK too
                sitk_msk = sitk.Cast(sitk_msk > 0, sitk.sitkUInt8)

                corrector = sitk.N4BiasFieldCorrectionImageFilter()
                corrector.SetMaximumNumberOfIterations(n4_num_iterations)
                corrector.SetConvergenceThreshold(n4_convergence_thresh)

                if n4_shrink_factor is not None and n4_shrink_factor > 1:
                    img_shr = sitk.Shrink(sitk_img, [n4_shrink_factor] * 3)
                    msk_shr = sitk.Shrink(sitk_msk, [n4_shrink_factor] * 3)
                    out_shr = corrector.Execute(img_shr, msk_shr)

                    # get bias field at full resolution and apply to full-res image
                    log_bias = corrector.GetLogBiasFieldAsImage(sitk_img)
                    sitk_corr = sitk_img / sitk.Exp(log_bias)
                else:
                    sitk_corr = corrector.Execute(sitk_img, sitk_msk)

                # save corrected image
                sitk.WriteImage(sitk_corr, out_path)

                # compute ROI stats "after"
                corr_np = nib.load(out_path).get_fdata().astype(np.float32)

                if corr_np.shape[:3] != msk_bin.shape[:3]:
                    print("Corrected/mask shape mismatch, skipping stats after:", folder_name)
                    continue

                mu_a, sd_a, cov_a = roi_stats(corr_np, msk_bin)

                # store per-subject row
                rows.append([
                    gname,
                    folder_name,
                    img_np.shape[0], img_np.shape[1], img_np.shape[2],
                    float(img_nii.header.get_zooms()[0]), float(img_nii.header.get_zooms()[1]), float(img_nii.header.get_zooms()[2]),
                    mu_b, sd_b, cov_b,
                    mu_a, sd_a, cov_a,
                    out_image_name
                ])

                group_before.append((mu_b, sd_b, cov_b))
                group_after.append((mu_a, sd_a, cov_a))
                all_before.append((mu_b, sd_b, cov_b))
                all_after.append((mu_a, sd_a, cov_a))

                print("Done:", folder_name, "| CoV before:", cov_b, "after:", cov_a)

        # per-group means
        def mean_triplets(trips):
            if len(trips) == 0:
                return (np.nan, np.nan, np.nan)
            arr = np.array(trips, dtype=np.float64)
            return (float(np.nanmean(arr[:, 0])), float(np.nanmean(arr[:, 1])), float(np.nanmean(arr[:, 2])))

        g_mu_b, g_sd_b, g_cov_b = mean_triplets(group_before)
        g_mu_a, g_sd_a, g_cov_a = mean_triplets(group_after)

        group_summaries.append([
            gname,
            len(group_before),
            g_mu_b, g_sd_b, g_cov_b,
            g_mu_a, g_sd_a, g_cov_a
        ])

    # overall means
    def mean_triplets(trips):
        if len(trips) == 0:
            return (np.nan, np.nan, np.nan)
        arr = np.array(trips, dtype=np.float64)
        return (float(np.nanmean(arr[:, 0])), float(np.nanmean(arr[:, 1])), float(np.nanmean(arr[:, 2])))

    all_mu_b, all_sd_b, all_cov_b = mean_triplets(all_before)
    all_mu_a, all_sd_a, all_cov_a = mean_triplets(all_after)

    # ------------------------
    # Write Excel
    # ------------------------
    wb = Workbook()

    # Sheet 1: per-image results
    ws = wb.active
    ws.title = "PerImage"

    headers = [
        "Group", "Folder",
        "X", "Y", "Z",
        "sx(mm)", "sy(mm)", "sz(mm)",
        "mu_before", "sd_before", "cov_before",
        "mu_after",  "sd_after",  "cov_after",
        "saved_corrected_filename"
    ]
    ws.append(headers)

    for r in rows:
        ws.append(r)

    # Sheet 2: group summary
    ws2 = wb.create_sheet("GroupSummary")
    ws2.append([
        "Group", "N_images",
        "mean_mu_before", "mean_sd_before", "mean_cov_before",
        "mean_mu_after",  "mean_sd_after",  "mean_cov_after"
    ])

    for s in group_summaries:
        ws2.append(s)

    # add overall row
    ws2.append([])
    ws2.append([
        "ALL_DATASETS", len(all_before),
        all_mu_b, all_sd_b, all_cov_b,
        all_mu_a, all_sd_a, all_cov_a
    ])

    wb.save(excel_out)
    print("\nSaved Excel:", excel_out)
    print("Overall mean CoV before:", all_cov_b, "after:", all_cov_a)


if __name__ == "__main__":
    main()
