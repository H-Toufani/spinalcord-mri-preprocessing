"""
Summary:
    Slice-wise CLAHE (Contrast Limited Adaptive Histogram Equalization) on cropped spinal cord MRI volumes
    + CNR evaluation using cord vs surrounding-tissue ring + Excel export.

What this script does:
    For each dataset group (multiple roots), it:
      1) Loads:
           - ORIGINAL (non-cropped) image (default: T2s.nii.gz) used only to estimate background sigma from corners
           - CROPPED image (default: T2s_zscore_norm.nii.gz) where CLAHE is applied
           - CROPPED cord mask (default: T2s_seg_cropped.nii.gz) aligned with cropped image
      2) Builds a "surrounding tissue" ring ROI by dilating the cord mask (inner/outer shells) and
         intersecting with a body mask (to exclude background/air).
      3) Computes CNR BEFORE CLAHE: |mu_cord - mu_surround| / sigma_bg
      4) Applies CLAHE slice-by-slice (2D CLAHE on each axial slice) on the cropped volume and saves output.
      5) Computes CNR AFTER CLAHE using the same ROIs and same sigma_bg.
      6) Exports per-case results + per-group + overall summary to an Excel file.

Requirements:
    pip install numpy nibabel scipy openpyxl scikit-image

Inputs (per subject folder):
    - Original image: T2s.nii.gz (configurable via --orig_image_name)
    - Cropped image : T2s_zscore_norm.nii.gz (configurable via --crop_image_name)
    - Cropped mask  : T2s_seg_cropped.nii.gz (configurable via --cord_mask_name)

Outputs:
    - CLAHE volume saved per subject folder (default: T2s_CLAHE.nii.gz)
    - Excel workbook (default: clahe_cnr_before_after_sigma_from_original.xlsx)

How to run:
    python 06_CLAHE_HE.py \
        --Site1_root /path/to/Site1_dataset \
        --Site2_root /path/to/Site2_dataset \
        --Site3_root  /path/to/Site3_dataset
"""

import os
import argparse
import numpy as np
import nibabel as nib
from scipy.ndimage import binary_dilation
from openpyxl import Workbook
from skimage.exposure import equalize_adapthist


def parse_args():
    parser = argparse.ArgumentParser(description="Slice-wise CLAHE + CNR evaluation + Excel export.")
    parser.add_argument("--Site1_root", type=str, default=os.environ.get("SITE1_ROOT", ""), help="Root directory for Site1.")
    parser.add_argument("--Site2_root", type=str, default=os.environ.get("SITE2_ROOT", ""), help="Root directory for Site2.")
    parser.add_argument("--Site3_root",  type=str, default=os.environ.get("SITE3_ROOT",  ""), help="Root directory for Site3.")

    parser.add_argument("--orig_image_name", type=str, default="T2s.nii.gz",
                        help="Original (non-cropped) image filename used only for sigma_bg estimation.")
    parser.add_argument("--crop_image_name", type=str, default="T2s_zscore_norm.nii.gz",
                        help="Cropped image filename where CLAHE is applied.")
    parser.add_argument("--cord_mask_name", type=str, default="T2s_seg_cropped.nii.gz",
                        help="Cropped cord mask filename aligned with cropped image.")

    parser.add_argument("--out_name", type=str, default="T2s_CLAHE.nii.gz", help="Output CLAHE filename saved per subject.")
    parser.add_argument("--excel_out", type=str, default="clahe_cnr_before_after_sigma_from_original.xlsx", help="Output Excel filename.")

    # CLAHE parameters
    parser.add_argument("--clip_limit", type=float, default=0.01, help="CLAHE clip limit (default: 0.01).")
    parser.add_argument("--kernel_size", type=int, nargs=2, default=[32, 32], help="CLAHE kernel/tile size (2D) (default: 32 32).")

    # Surrounding tissue ring parameters
    parser.add_argument("--ring_inner", type=int, default=2)
    parser.add_argument("--ring_outer", type=int, default=5)

    # Body threshold in cropped space
    parser.add_argument("--body_threshold_percentile", type=float, default=20.0)

    # Background sigma corner size (original image)
    parser.add_argument("--bg_corner_size", type=int, default=30)

    return parser.parse_args()


EPS = 1e-8


# -----------------------------
# HELPERS
# -----------------------------
def nanmean(x):
    return float(np.nanmean(np.array(x, dtype=np.float64)))


def pct_improve(before, after):
    if np.isnan(before) or abs(before) < EPS:
        return np.nan
    return (after - before) / abs(before) * 100.0


def background_sigma_from_corners_robust(vol3d, corner=30):
    """
    Robust background sigma from ORIGINAL image:
    - take corner voxels (all Z)
    - select low-intensity subset using global percentiles
    - fall back safely if needed
    """
    x, y, z = vol3d.shape
    c = min(corner, x, y)

    m2d = np.zeros((x, y), dtype=bool)
    m2d[:c, :c] = True
    m2d[:c, y - c:y] = True
    m2d[x - c:x, :c] = True
    m2d[x - c:x, y - c:y] = True

    corner_vals = vol3d[m2d, :].astype(np.float64).ravel()
    if corner_vals.size < 50:
        return float(np.std(corner_vals)) if corner_vals.size > 0 else np.nan

    global_vals = vol3d.astype(np.float64).ravel()

    for p in [5, 10, 20, 30]:
        thr = np.percentile(global_vals, p)
        air_like = corner_vals[corner_vals <= thr]
        if air_like.size >= 200:
            sigma = float(np.std(air_like))
            if sigma > 0:
                return sigma

    return float(np.std(corner_vals))


def surrounding_ring_mask(cord_mask, vol3d, inner=2, outer=5, body_thresh_percentile=20):
    """
    Approximate 'surrounding tissues' as a ring around the cord.
    Intersect with a body mask to avoid air/background.
    """
    dil_outer = binary_dilation(cord_mask, iterations=outer)
    dil_inner = binary_dilation(cord_mask, iterations=inner)
    ring = np.logical_and(dil_outer, np.logical_not(dil_inner))

    thr = np.percentile(vol3d, body_thresh_percentile)
    body = vol3d > thr
    return np.logical_and(ring, body)


def apply_clahe_3d_slicewise(vol3d, clip_limit=0.01, kernel_size=(32, 32)):
    """
    Apply CLAHE per slice along Z (2D CLAHE on each slice).
    - Normalize each slice to [0,1] using robust percentiles
    - Apply CLAHE
    - Map back to original slice intensity range
    """
    x, y, z = vol3d.shape
    out = np.zeros_like(vol3d, dtype=np.float32)

    for k in range(z):
        sl = vol3d[:, :, k].astype(np.float32)

        p1 = np.percentile(sl, 1)
        p99 = np.percentile(sl, 99)
        denom = (p99 - p1) if (p99 - p1) > EPS else 1.0

        sl01 = (sl - p1) / denom
        sl01 = np.clip(sl01, 0.0, 1.0)

        eq = equalize_adapthist(sl01, clip_limit=clip_limit, kernel_size=kernel_size)

        out[:, :, k] = (eq.astype(np.float32) * denom + p1)

    return out


# -----------------------------
# MAIN
# -----------------------------
def main():
    args = parse_args()

    GROUPS = {
        "Site1": args.Site1_root,
        "Site2": args.Site2_root,
        "Site3": args.Site3_root,
    }

    ORIG_IMAGE_NAME = args.orig_image_name
    CROP_IMAGE_NAME = args.crop_image_name
    CORD_MASK_NAME = args.cord_mask_name
    OUT_NAME = args.out_name

    CLIP_LIMIT = args.clip_limit
    KERNEL_SIZE = (int(args.kernel_size[0]), int(args.kernel_size[1]))

    RING_INNER = args.ring_inner
    RING_OUTER = args.ring_outer
    BODY_THRESHOLD_PERCENTILE = args.body_threshold_percentile
    BG_CORNER_SIZE = args.bg_corner_size

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
            "Site1, Site2, Site3."
        )

    results = []

    for group_name, group_root in valid_groups.items():
        for root, dirs, files in os.walk(group_root):
            if CROP_IMAGE_NAME in files:
                folder_name = os.path.basename(root)

                orig_path = os.path.join(root, ORIG_IMAGE_NAME)
                crop_path = os.path.join(root, CROP_IMAGE_NAME)
                mask_path = os.path.join(root, CORD_MASK_NAME)

                if not os.path.exists(orig_path):
                    print("Skipping (missing original image):", group_name, folder_name)
                    continue
                if not os.path.exists(mask_path):
                    print("Skipping (missing cropped mask):", group_name, folder_name)
                    continue

                # ---- sigma_background from ORIGINAL image ----
                orig_vol = nib.load(orig_path).get_fdata().astype(np.float32)
                sigma_bg = background_sigma_from_corners_robust(orig_vol, corner=BG_CORNER_SIZE)

                if np.isnan(sigma_bg) or sigma_bg < EPS:
                    print("Skipping (sigma_bg invalid):", group_name, folder_name)
                    continue
                sigma_bg = float(sigma_bg)

                # ---- Load CROPPED image + mask ----
                crop_img = nib.load(crop_path)
                vol = crop_img.get_fdata().astype(np.float32)
                affine = crop_img.affine
                spacing = crop_img.header.get_zooms()[:3]

                cord = nib.load(mask_path).get_fdata().astype(np.float32)
                cord_mask = cord > 0.5

                # Surrounding tissues ring in CROPPED space
                surround_mask = surrounding_ring_mask(
                    cord_mask=cord_mask,
                    vol3d=vol,
                    inner=RING_INNER,
                    outer=RING_OUTER,
                    body_thresh_percentile=BODY_THRESHOLD_PERCENTILE
                )

                cord_vals_b = vol[cord_mask]
                sur_vals_b = vol[surround_mask]

                if cord_vals_b.size == 0 or sur_vals_b.size == 0:
                    print("Skipping (empty ROI before):", group_name, folder_name,
                          "cord:", cord_vals_b.size, "sur:", sur_vals_b.size)
                    continue

                mu_cord_b = float(np.mean(cord_vals_b))
                mu_sur_b = float(np.mean(sur_vals_b))
                cnr_before = abs(mu_cord_b - mu_sur_b) / sigma_bg

                # ---- CLAHE on CROPPED image ----
                vol_clahe = apply_clahe_3d_slicewise(
                    vol,
                    clip_limit=CLIP_LIMIT,
                    kernel_size=KERNEL_SIZE
                )

                out_path = os.path.join(root, OUT_NAME)
                nib.save(nib.Nifti1Image(vol_clahe, affine), out_path)

                # ---- After CNR (same ROIs, same sigma_bg) ----
                cord_vals_a = vol_clahe[cord_mask]
                sur_vals_a = vol_clahe[surround_mask]

                if cord_vals_a.size == 0 or sur_vals_a.size == 0:
                    print("Skipping (empty ROI after):", group_name, folder_name)
                    continue

                mu_cord_a = float(np.mean(cord_vals_a))
                mu_sur_a = float(np.mean(sur_vals_a))
                cnr_after = abs(mu_cord_a - mu_sur_a) / sigma_bg
                cnr_imp = pct_improve(cnr_before, cnr_after)

                results.append({
                    "Group": group_name,
                    "Folder": folder_name,
                    "X": int(vol.shape[0]),
                    "Y": int(vol.shape[1]),
                    "Z": int(vol.shape[2]),
                    "sx(mm)": float(spacing[0]),
                    "sy(mm)": float(spacing[1]),
                    "sz(mm)": float(spacing[2]),
                    "sigma_background_from_original": sigma_bg,

                    "mu_cord_before": mu_cord_b,
                    "mu_surround_before": mu_sur_b,
                    "CNR_before": float(cnr_before),

                    "mu_cord_after": mu_cord_a,
                    "mu_surround_after": mu_sur_a,
                    "CNR_after": float(cnr_after),

                    "CNR_%improvement": float(cnr_imp) if cnr_imp == cnr_imp else np.nan,
                    "CLAHE_clip_limit": CLIP_LIMIT,
                    "CLAHE_kernel": str(KERNEL_SIZE),
                    "OutFile": OUT_NAME
                })

                print("Saved:", out_path, "|", group_name, folder_name,
                      "| CNR:", cnr_before, "->", cnr_after, "| %:", cnr_imp)

    if len(results) == 0:
        raise ValueError("No cases processed. Check filenames and masks.")

    # -----------------------------
    # SUMMARY (means per group + overall)
    # -----------------------------
    summary_rows = []
    for scope in list(valid_groups.keys()) + ["ALL_DATASET"]:
        if scope == "ALL_DATASET":
            items = results
        else:
            items = [r for r in results if r["Group"] == scope]

        cnr_b = [r["CNR_before"] for r in items]
        cnr_a = [r["CNR_after"] for r in items]
        cnr_i = [r["CNR_%improvement"] for r in items]

        summary_rows.append({
            "Scope": scope,
            "N": len(items),
            "Mean_CNR_before": nanmean(cnr_b),
            "Mean_CNR_after": nanmean(cnr_a),
            "Mean_CNR_%improvement": nanmean(cnr_i),
        })

    # -----------------------------
    # SAVE EXCEL
    # -----------------------------
    wb = Workbook()
    ws = wb.active
    ws.title = "PerImage"

    headers = [
        "Group", "Folder",
        "X", "Y", "Z",
        "sx(mm)", "sy(mm)", "sz(mm)",
        "sigma_background_from_original",
        "mu_cord_before", "mu_surround_before", "CNR_before",
        "mu_cord_after", "mu_surround_after", "CNR_after",
        "CNR_%improvement",
        "CLAHE_clip_limit", "CLAHE_kernel",
        "OutFile"
    ]
    ws.append(headers)

    for r in results:
        ws.append([
            r["Group"], r["Folder"],
            r["X"], r["Y"], r["Z"],
            r["sx(mm)"], r["sy(mm)"], r["sz(mm)"],
            r["sigma_background_from_original"],
            r["mu_cord_before"], r["mu_surround_before"], r["CNR_before"],
            r["mu_cord_after"], r["mu_surround_after"], r["CNR_after"],
            r["CNR_%improvement"],
            r["CLAHE_clip_limit"], r["CLAHE_kernel"],
            r["OutFile"]
        ])

    ws2 = wb.create_sheet("Summary")
    ws2.append(["Scope", "N", "Mean_CNR_before", "Mean_CNR_after", "Mean_CNR_%improvement"])
    for s in summary_rows:
        ws2.append([s["Scope"], s["N"], s["Mean_CNR_before"], s["Mean_CNR_after"], s["Mean_CNR_%improvement"]])

    out_xlsx = args.excel_out
    wb.save(out_xlsx)

    print("Saved Excel:", out_xlsx)
    print("Total images:", len(results))


if __name__ == "__main__":
    main()
