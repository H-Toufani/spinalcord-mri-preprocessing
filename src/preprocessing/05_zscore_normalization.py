"""
Summary:
    Z-score normalization of cropped spinal cord MRI volumes + inter-subject variance analysis + Excel export.

What this script does:
    For each dataset group (multiple roots), it:
      1) Finds subject folders containing the input image (default: T2s_nlm_denoised.nii.gz).
      2) Builds a simple "body" mask using a low percentile threshold to exclude air/background.
      3) Computes mean and std within the body mask (BEFORE).
      4) Applies z-score normalization using those body stats, but applies the transform to ALL voxels.
      5) Computes mean and std within the same body mask (AFTER).
      6) Saves the normalized image per subject folder (default: T2s_zscore_norm.nii.gz).
      7) Computes inter-subject variance using the variance of per-subject body means:
           - per group
           - overall across all datasets
      8) Exports per-image metrics and summary statistics to Excel.

Requirements:
    pip install numpy nibabel openpyxl

Inputs (per subject folder):
    - Image: T2s_nlm_denoised.nii.gz (configurable via --image_name)

Outputs:
    - Normalized image saved per subject folder: T2s_zscore_norm.nii.gz (configurable via --out_name)
    - Excel workbook: zscore_intersubject_variance.xlsx (configurable via --excel_out)

How to run:
    python 05-zscore_normalization.py \
        --Site1_root /path/to/Site1_dataset \
        --Site2_root /path/to/Site2_dataset \
        --Site3_root  /path/to/Site3_dataset

"""

import os
import argparse
import numpy as np
import nibabel as nib
from openpyxl import Workbook


def parse_args():
    parser = argparse.ArgumentParser(description="Z-score normalization + inter-subject variance analysis + Excel export.")
    parser.add_argument("--Site1_root", type=str, default=os.environ.get("SITE1_ROOT", ""), help="Root directory for Site1.")
    parser.add_argument("--Site2_root", type=str, default=os.environ.get("SITE2_ROOT", ""), help="Root directory for Site2.")
    parser.add_argument("--Site3_root",  type=str, default=os.environ.get("SITE3_ROOT",  ""), help="Root directory for Site3.")

    parser.add_argument("--image_name", type=str, default="T2s_nlm_denoised.nii.gz", help="Input image filename to normalize.")
    parser.add_argument("--out_name", type=str, default="T2s_zscore_norm.nii.gz", help="Output normalized filename saved per subject folder.")
    parser.add_argument("--body_percentile", type=float, default=20.0, help="Percentile threshold for body mask (default: 20).")
    parser.add_argument("--excel_out", type=str, default="zscore_intersubject_variance.xlsx", help="Output Excel filename.")
    return parser.parse_args()


EPS = 1e-8


# -----------------------------
# HELPERS
# -----------------------------
def nanmean(x):
    return float(np.nanmean(np.array(x, dtype=np.float64)))


def nanvar(x):
    return float(np.nanvar(np.array(x, dtype=np.float64)))


def pct_change(before, after):
    # percent change in variance: (after - before)/before * 100
    if np.isnan(before) or abs(before) < EPS:
        return np.nan
    return (after - before) / abs(before) * 100.0


def main():
    args = parse_args()

    GROUPS = {
        "Site1": args.Site1_root,
        "Site2": args.Site2_root,
        "Site3": args.Site3_root,
    }

    IMAGE_NAME = args.image_name
    OUT_NAME = args.out_name
    BODY_PERCENTILE = args.body_percentile

    # Validate roots (skip empty/invalid to allow running on one dataset)
    valid_groups = {}
    for name, root in GROUPS.items():
        if root and os.path.isdir(root):
            valid_groups[name] = root
        else:
            if root != "":
                print(f'Warning: root not found or invalid for group "{name}": {root}')

    if len(valid_groups) == 0:
        raise ValueError(
            "No valid dataset roots provided. Use --Site1_root/--Site2_root/--Site3_root or set env vars "
            "Site1_ROOT, Site2_ROOT, Site3_ROOT."
        )

    # -----------------------------
    # MAIN
    # -----------------------------
    rows = []  # per-image results

    for group_name, group_root in valid_groups.items():
        for root, dirs, files in os.walk(group_root):
            if IMAGE_NAME in files:
                folder_name = os.path.basename(root)
                in_path = os.path.join(root, IMAGE_NAME)

                img = nib.load(in_path)
                vol = img.get_fdata().astype(np.float32)
                affine = img.affine
                header = img.header

                # "Body" mask to avoid background air
                thr = np.percentile(vol, BODY_PERCENTILE)
                body = vol > thr

                if np.count_nonzero(body) < 50:
                    # fallback: use all voxels if body mask is too small
                    body = np.ones_like(vol, dtype=bool)

                # Before stats (within body)
                mu_before = float(np.mean(vol[body]))
                sd_before = float(np.std(vol[body]))

                # Z-score normalization (within body) applied to ALL voxels
                denom = sd_before if sd_before > EPS else 1.0
                vol_z = (vol - mu_before) / denom
                vol_z = vol_z.astype(np.float32)

                # After stats (within same body mask)
                mu_after = float(np.mean(vol_z[body]))
                sd_after = float(np.std(vol_z[body]))

                # Save normalized image
                out_path = os.path.join(root, OUT_NAME)
                nib.save(nib.Nifti1Image(vol_z, affine, header=header), out_path)

                rows.append({
                    "Group": group_name,
                    "Folder": folder_name,
                    "X": int(vol.shape[0]),
                    "Y": int(vol.shape[1]),
                    "Z": int(vol.shape[2]),
                    "BodyPercentile": BODY_PERCENTILE,
                    "BodyVoxels": int(np.count_nonzero(body)),
                    "Mean_before": mu_before,
                    "Std_before": sd_before,
                    "Mean_after": mu_after,
                    "Std_after": sd_after,
                    "OutFile": OUT_NAME
                })

                print("Saved:", out_path, "|", group_name, folder_name)

    if len(rows) == 0:
        raise ValueError("No images processed. Check dataset roots and IMAGE_NAME.")

    # -----------------------------
    # INTER-SUBJECT VARIANCE
    # -----------------------------
    # We quantify between-subject intensity variance using the variance of per-subject body means.
    # Compute per group and overall:
    summary = []

    for scope in list(valid_groups.keys()) + ["ALL_DATASET"]:
        if scope == "ALL_DATASET":
            items = rows
        else:
            items = [r for r in rows if r["Group"] == scope]

        means_before = [r["Mean_before"] for r in items]
        means_after = [r["Mean_after"] for r in items]
        stds_before = [r["Std_before"] for r in items]
        stds_after = [r["Std_after"] for r in items]

        var_before = nanvar(means_before)
        var_after = nanvar(means_after)

        summary.append({
            "Scope": scope,
            "N": len(items),
            "InterSubjectVar_Mean_before": var_before,
            "InterSubjectVar_Mean_after": var_after,
            "Var_%change": pct_change(var_before, var_after),
            "Mean_of_Std_before": nanmean(stds_before),
            "Mean_of_Std_after": nanmean(stds_after),
        })

    # -----------------------------
    # SAVE EXCEL
    # -----------------------------
    wb = Workbook()

    ws = wb.active
    ws.title = "PerImage"

    headers = [
        "Group", "Folder", "X", "Y", "Z",
        "BodyPercentile", "BodyVoxels",
        "Mean_before", "Std_before",
        "Mean_after", "Std_after",
        "OutFile"
    ]
    ws.append(headers)

    for r in rows:
        ws.append([
            r["Group"], r["Folder"], r["X"], r["Y"], r["Z"],
            r["BodyPercentile"], r["BodyVoxels"],
            r["Mean_before"], r["Std_before"],
            r["Mean_after"], r["Std_after"],
            r["OutFile"]
        ])

    ws2 = wb.create_sheet("Summary")
    ws2.append([
        "Scope", "N",
        "InterSubjectVar_Mean_before", "InterSubjectVar_Mean_after", "Var_%change",
        "Mean_of_Std_before", "Mean_of_Std_after"
    ])

    for s in summary:
        ws2.append([
            s["Scope"], s["N"],
            s["InterSubjectVar_Mean_before"], s["InterSubjectVar_Mean_after"], s["Var_%change"],
            s["Mean_of_Std_before"], s["Mean_of_Std_after"]
        ])

    out_xlsx = args.excel_out
    wb.save(out_xlsx)

    print("Saved Excel:", out_xlsx)
    print("Total images:", len(rows))


if __name__ == "__main__":
    main()
