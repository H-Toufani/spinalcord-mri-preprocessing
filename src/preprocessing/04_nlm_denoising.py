"""
Summary:
    Non-Local Means (NLM) denoising on spinal cord MRI volumes + SNR/CNR evaluation + Excel export.

What this script does:
    For each dataset group (multiple roots), it:
      1) Loads:
           - Cropped input image (default: T2s_N4_bias_field_correction.nii.gz)
           - Original image (default: T2s.nii.gz) used only to estimate background noise sigma
           - Cropped cord mask (default: T2s_seg_cropped.nii.gz) aligned with the cropped image
      2) Computes a CSF "ring" ROI by dilating the cord mask (inner/outer shells) and
         excluding low-intensity background/body voxels.
      3) Computes SNR and CNR BEFORE denoising on the cropped image, using sigma_bg from the original image corners.
      4) Runs DIPY NLM denoising on the cropped image and saves the denoised output per subject.
      5) Computes SNR and CNR AFTER denoising using the same ROIs and the same sigma_bg denominator.
      6) Exports per-case results + per-group + overall summary to an Excel file.

Requirements:
    pip install numpy nibabel dipy scipy openpyxl

Inputs (per subject folder):
    - Cropped image : (default) T2s_N4_bias_field_correction.nii.gz
    - Original image: (default) T2s.nii.gz
    - Cropped cord mask (ROI): (default) T2s_seg_cropped.nii.gz

Outputs:
    - Denoised volume saved per subject folder (default: T2s_nlm_denoised.nii.gz)
    - Excel workbook (default: 04_nlm_snr_cnr_before_after_improvement.xlsx)

How to run:
    python 04_nlm_denoise.py \
        --Site1_root /path/to/Site1_dataset \
        --Site2_root /path/to/Site2_dataset \
        --Site3_root  /path/to/Site3_dataset
"""

import os
import argparse
import numpy as np
import nibabel as nib

from dipy.denoise.nlmeans import nlmeans
from dipy.denoise.noise_estimate import estimate_sigma
from scipy.ndimage import binary_dilation
from openpyxl import Workbook


def parse_args():
    parser = argparse.ArgumentParser(description="NLM denoising + SNR/CNR evaluation + Excel export.")
    parser.add_argument("--Site1_root", type=str, default=os.environ.get("SITE1_ROOT", ""), help="Root directory for Site1.")
    parser.add_argument("--Site2_root", type=str, default=os.environ.get("SITE2_ROOT", ""), help="Root directory for Site2.")
    parser.add_argument("--Site3_root",  type=str, default=os.environ.get("SITE3_ROOT",  ""), help="Root directory for Site3.")

    parser.add_argument("--cropped_image_name", type=str, default="T2s_N4_bias_field_correction.nii.gz",
                        help="Cropped input image filename used for denoising + metrics.")
    parser.add_argument("--original_image_name", type=str, default="T2s.nii.gz",
                        help="Original (non-cropped) image filename used only for sigma_bg estimation.")
    parser.add_argument("--cord_mask_name", type=str, default="T2s_seg_cropped.nii.gz",
                        help="Cropped cord mask filename aligned with the cropped image.")
    parser.add_argument("--out_denoised_name", type=str, default="T2s_N4_nlm_denoised.nii.gz",
                        help="Output denoised filename saved per subject folder.")
    parser.add_argument("--excel_out", type=str, default="nlm_snr_cnr_before_after_improvement.xlsx",
                        help="Output Excel filename.")

    # NLM parameters (kept as in your script)
    parser.add_argument("--patch_radius", type=int, default=1)
    parser.add_argument("--block_radius", type=int, default=3)
    parser.add_argument("--rician", action="store_true", help="Use Rician noise model (default: True in original script).")

    # CSF ring parameters
    parser.add_argument("--csf_ring_inner", type=int, default=2)
    parser.add_argument("--csf_ring_outer", type=int, default=5)

    # Background (noise) ROI from corners
    parser.add_argument("--bg_corner_size", type=int, default=20)

    # Exclude background from CSF ring
    parser.add_argument("--body_threshold_percentile", type=float, default=20.0)

    return parser.parse_args()


EPS = 1e-8


# -----------------------------
# HELPERS
# -----------------------------
def background_sigma_from_corners_air(vol3d, corner=20):
    """
    Robust background sigma estimator for ORIGINAL images.
    Works even if corners are not pure air.

    Strategy:
    - Take corner voxels (all Z)
    - Keep "air-like" voxels using a global low-intensity threshold
      (try 5th/10th/20th/30th percentile of whole volume), relax if needed.
    - Fall back safely if selection becomes too small.
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

    # Fallback: use all corner voxels
    sigma = float(np.std(corner_vals))
    return sigma


def csf_ring_mask(cord_mask, vol3d, inner=2, outer=5, body_thresh_percentile=20):
    dil_outer = binary_dilation(cord_mask, iterations=outer)
    dil_inner = binary_dilation(cord_mask, iterations=inner)
    ring = np.logical_and(dil_outer, np.logical_not(dil_inner))

    thr = np.percentile(vol3d, body_thresh_percentile)
    body = vol3d > thr

    return np.logical_and(ring, body)


def nanmean(x):
    return float(np.nanmean(np.array(x, dtype=np.float64)))


def pct_improve(before, after):
    if before is None or after is None:
        return np.nan
    if np.isnan(before) or np.isnan(after) or abs(before) < EPS:
        return np.nan
    return (after - before) / abs(before) * 100.0


# -----------------------------
# MAIN
# -----------------------------
def main():
    args = parse_args()

    # Default behavior in your original script: RICIAN=True
    # Here, if user does not pass --rician, we still keep it True to match original behavior.
    # (Using action="store_true" defaults to False; so we override here.)
    RICIAN = True

    GROUPS = {
        "Site1": args.Site1_root,
        "Site2": args.Site2_root,
        "Site3": args.Site3_root,
    }

    CROPPED_IMAGE_NAME = args.cropped_image_name
    ORIGINAL_IMAGE_NAME = args.original_image_name
    CORD_MASK_NAME = args.cord_mask_name
    OUT_DENOISED_NAME = args.out_denoised_name

    PATCH_RADIUS = args.patch_radius
    BLOCK_RADIUS = args.block_radius

    CSF_RING_INNER = args.csf_ring_inner
    CSF_RING_OUTER = args.csf_ring_outer

    BG_CORNER_SIZE = args.bg_corner_size
    BODY_THRESHOLD_PERCENTILE = args.body_threshold_percentile

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
            "SITE1_ROOT, SITE2_ROOT, SITE3_ROOT."
        )

    results = []

    for group_name, group_root in valid_groups.items():
        for root, dirs, files in os.walk(group_root):
            if CROPPED_IMAGE_NAME in files:
                folder_name = os.path.basename(root)

                cropped_path = os.path.join(root, CROPPED_IMAGE_NAME)
                original_path = os.path.join(root, ORIGINAL_IMAGE_NAME)
                mask_path = os.path.join(root, CORD_MASK_NAME)

                if not os.path.exists(original_path):
                    print("Skipping (missing ORIGINAL image):", group_name, folder_name)
                    continue
                if not os.path.exists(mask_path):
                    print("Skipping (missing CROPPED mask):", group_name, folder_name)
                    continue

                print("Processing:", group_name, "|", folder_name)

                # --- Read original image ONLY for sigma_background ---
                orig_img = nib.load(original_path)
                orig_vol = orig_img.get_fdata().astype(np.float32)

                sigma_bg = background_sigma_from_corners_air(
                    orig_vol,
                    corner=BG_CORNER_SIZE,
                )
                if np.isnan(sigma_bg) or sigma_bg < EPS:
                    print("  Warning: sigma_bg invalid -> skipping:", folder_name)
                    continue
                sigma_bg = max(float(sigma_bg), EPS)

                # --- Read cropped image for denoising + metrics ---
                img = nib.load(cropped_path)
                vol = img.get_fdata().astype(np.float32)
                affine = img.affine
                spacing = img.header.get_zooms()[:3]

                cord_img = nib.load(mask_path)
                cord = cord_img.get_fdata()
                cord_mask = cord > 0.5

                # CSF mask is based on CROPPED image geometry
                csf_mask = csf_ring_mask(
                    cord_mask=cord_mask,
                    vol3d=vol,
                    inner=CSF_RING_INNER,
                    outer=CSF_RING_OUTER,
                    body_thresh_percentile=BODY_THRESHOLD_PERCENTILE
                )

                # ---- BEFORE metrics (on CROPPED original, denominator from ORIGINAL sigma_bg) ----
                cord_vals_b = vol[cord_mask]
                csf_vals_b = vol[csf_mask]

                if cord_vals_b.size == 0 or csf_vals_b.size == 0:
                    print("  Warning: empty ROI before -> skipping:", folder_name)
                    continue

                mu_cord_b = float(np.mean(cord_vals_b))
                mu_csf_b = float(np.mean(csf_vals_b))
                snr_before = mu_cord_b / sigma_bg
                cnr_before = abs(mu_cord_b - mu_csf_b) / sigma_bg

                # ---- DENOISING (KEEP EXACTLY AS BEFORE) ----
                sigma_est = float(estimate_sigma(vol, N=1))

                vol_nlm = nlmeans(
                    vol,
                    sigma=sigma_est,
                    patch_radius=PATCH_RADIUS,
                    block_radius=BLOCK_RADIUS,
                    rician=RICIAN
                ).astype(np.float32)

                # Save denoised output per case
                out_path = os.path.join(root, OUT_DENOISED_NAME)
                nib.save(nib.Nifti1Image(vol_nlm, affine), out_path)

                # ---- AFTER metrics (on denoised CROPPED, denominator from ORIGINAL sigma_bg) ----
                cord_vals_a = vol_nlm[cord_mask]
                csf_vals_a = vol_nlm[csf_mask]

                if cord_vals_a.size == 0 or csf_vals_a.size == 0:
                    print("  Warning: empty ROI after -> skipping:", folder_name)
                    continue

                mu_cord_a = float(np.mean(cord_vals_a))
                mu_csf_a = float(np.mean(csf_vals_a))
                snr_after = mu_cord_a / sigma_bg
                cnr_after = abs(mu_cord_a - mu_csf_a) / sigma_bg

                snr_impr = pct_improve(snr_before, snr_after)
                cnr_impr = pct_improve(cnr_before, cnr_after)

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
                    "sigma_est_for_nlm_on_cropped": sigma_est,

                    "mu_cord_before": mu_cord_b,
                    "mu_csf_before": mu_csf_b,
                    "SNR_before": float(snr_before),
                    "CNR_before": float(cnr_before),

                    "mu_cord_after": mu_cord_a,
                    "mu_csf_after": mu_csf_a,
                    "SNR_after": float(snr_after),
                    "CNR_after": float(cnr_after),

                    "SNR_%improvement": float(snr_impr) if snr_impr == snr_impr else np.nan,
                    "CNR_%improvement": float(cnr_impr) if cnr_impr == cnr_impr else np.nan,

                    "DenoisedFile": OUT_DENOISED_NAME
                })

                print("  Saved:", out_path)
                print("  sigma_bg(original):", sigma_bg)
                print("  SNR:", snr_before, "->", snr_after, "| %:", snr_impr)
                print("  CNR:", cnr_before, "->", cnr_after, "| %:", cnr_impr)
                print("-" * 60)

    if len(results) == 0:
        raise ValueError("No cases processed. Check dataset roots and filenames.")

    # -----------------------------
    # SUMMARY
    # -----------------------------
    summary_rows = []
    for g in valid_groups.keys():
        snr_b = [r["SNR_before"] for r in results if r["Group"] == g]
        snr_a = [r["SNR_after"] for r in results if r["Group"] == g]
        cnr_b = [r["CNR_before"] for r in results if r["Group"] == g]
        cnr_a = [r["CNR_after"] for r in results if r["Group"] == g]

        snr_imp = [r["SNR_%improvement"] for r in results if r["Group"] == g]
        cnr_imp = [r["CNR_%improvement"] for r in results if r["Group"] == g]

        summary_rows.append({
            "Scope": g,
            "N": len(snr_b),
            "Mean_SNR_before": nanmean(snr_b),
            "Mean_SNR_after": nanmean(snr_a),
            "Mean_SNR_%improvement": nanmean(snr_imp),
            "Mean_CNR_before": nanmean(cnr_b),
            "Mean_CNR_after": nanmean(cnr_a),
            "Mean_CNR_%improvement": nanmean(cnr_imp),
        })

    snr_b_all = [r["SNR_before"] for r in results]
    snr_a_all = [r["SNR_after"] for r in results]
    cnr_b_all = [r["CNR_before"] for r in results]
    cnr_a_all = [r["CNR_after"] for r in results]
    snr_imp_all = [r["SNR_%improvement"] for r in results]
    cnr_imp_all = [r["CNR_%improvement"] for r in results]

    summary_rows.append({
        "Scope": "ALL_DATASET",
        "N": len(snr_b_all),
        "Mean_SNR_before": nanmean(snr_b_all),
        "Mean_SNR_after": nanmean(snr_a_all),
        "Mean_SNR_%improvement": nanmean(snr_imp_all),
        "Mean_CNR_before": nanmean(cnr_b_all),
        "Mean_CNR_after": nanmean(cnr_a_all),
        "Mean_CNR_%improvement": nanmean(cnr_imp_all),
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
        "sigma_est_for_nlm_on_cropped",

        "mu_cord_before", "mu_csf_before", "SNR_before", "CNR_before",
        "mu_cord_after", "mu_csf_after", "SNR_after", "CNR_after",

        "SNR_%improvement", "CNR_%improvement",
        "DenoisedFile"
    ]
    ws.append(headers)

    for r in results:
        ws.append([
            r["Group"], r["Folder"],
            r["X"], r["Y"], r["Z"],
            r["sx(mm)"], r["sy(mm)"], r["sz(mm)"],
            r["sigma_background_from_original"],
            r["sigma_est_for_nlm_on_cropped"],

            r["mu_cord_before"], r["mu_csf_before"], r["SNR_before"], r["CNR_before"],
            r["mu_cord_after"], r["mu_csf_after"], r["SNR_after"], r["CNR_after"],

            r["SNR_%improvement"], r["CNR_%improvement"],
            r["DenoisedFile"]
        ])

    ws2 = wb.create_sheet("Summary")
    ws2.append([
        "Scope", "N",
        "Mean_SNR_before", "Mean_SNR_after", "Mean_SNR_%improvement",
        "Mean_CNR_before", "Mean_CNR_after", "Mean_CNR_%improvement"
    ])
    for s in summary_rows:
        ws2.append([
            s["Scope"], s["N"],
            s["Mean_SNR_before"], s["Mean_SNR_after"], s["Mean_SNR_%improvement"],
            s["Mean_CNR_before"], s["Mean_CNR_after"], s["Mean_CNR_%improvement"]
        ])

    out_xlsx = args.excel_out
    wb.save(out_xlsx)

    print("Saved Excel:", out_xlsx)
    print("Total cases:", len(results))


if __name__ == "__main__":
    main()
