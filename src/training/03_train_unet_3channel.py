"""
train_unet_monai_3ch_aug_no_earlystop.py

3D U-Net (MONAI) with 3 input channels
- Train on ALL datasets
- Random split: 70% train, 15% val, 15% test
- Augmentation (spatial + intensity) on TRAIN only
- Train 50 epochs (NO early stopping)
- Save best checkpoint (highest val Dice) + final model
- Test metrics: Dice, Precision, Recall
- Plot loss and Dice curves (train/val)

IMPORTANT:
- CHANNEL_FILES and MASK_NAME must exist in each case folder
- Mask must be aligned with the image (same crop space)
"""

import os
import random
import argparse
import numpy as np
import nibabel as nib
import torch
import matplotlib.pyplot as plt
import multiprocessing as mp

from torch.utils.data import Dataset, DataLoader

from monai.networks.nets import UNet
from monai.losses import DiceLoss
from monai.inferers import SimpleInferer
from monai.transforms import (
    Compose,
    RandScaleIntensityd,
    RandShiftIntensityd,
    RandGaussianNoised,
    RandAffined,
    Rand3DElasticd,
)

# ----------------------------
# SETTINGS
# ----------------------------
# Default to env vars, but can also be passed via CLI arguments.
GROUPS = {
    "Site1": os.environ.get("Site1_ROOT", ""),
    "Site2": os.environ.get("Site2_ROOT", ""),
    "Site3": os.environ.get("Site3_ROOT", ""),
}

CHANNEL_FILES = [
    "T2s_zscore_norm.nii.gz",   # Channel 1
    "T2s_CLAHE.nii.gz",         # Channel 2
    "T2s_LoG.nii.gz",       # Channel 3
]

MASK_NAME = "T2s_seg_cropped.nii.gz"

BEST_MODEL_PATH = "unet_3ch_best.pth"
FINAL_MODEL_PATH = "unet_3ch_final.pth"
LOSS_PLOT_PATH = "unet_3ch_loss_curve.png"
DICE_PLOT_PATH = "unet_3ch_dice_curve.png"

SEED = 42
MAX_EPOCHS = 50
LR = 1e-3

BATCH_SIZE = 4
NUM_WORKERS = 2
PIN_MEMORY = False

THRESH = 0.5


def parse_args():
    parser = argparse.ArgumentParser(description="3D U-Net (MONAI) training with 3 input channels + augmentation.")
    parser.add_argument("--Site1_root", type=str, default=GROUPS["Site1"], help="Root directory for Site1.")
    parser.add_argument("--Site2_root", type=str, default=GROUPS["Site2"], help="Root directory for Site2.")
    parser.add_argument("--Site3_root",  type=str, default=GROUPS["Site3"], help="Root directory for Site3.")

    parser.add_argument(
        "--channel_files",
        type=str,
        nargs=3,
        default=CHANNEL_FILES,
        help="Three filenames used as input channels (must exist in each case folder)."
    )
    parser.add_argument("--mask_name", type=str, default=MASK_NAME, help="Mask filename in each case folder.")

    parser.add_argument("--best_model_path", type=str, default=BEST_MODEL_PATH)
    parser.add_argument("--final_model_path", type=str, default=FINAL_MODEL_PATH)
    parser.add_argument("--loss_plot_path", type=str, default=LOSS_PLOT_PATH)
    parser.add_argument("--dice_plot_path", type=str, default=DICE_PLOT_PATH)

    parser.add_argument("--seed", type=int, default=SEED)
    parser.add_argument("--max_epochs", type=int, default=MAX_EPOCHS)
    parser.add_argument("--lr", type=float, default=LR)

    parser.add_argument("--batch_size", type=int, default=BATCH_SIZE)
    parser.add_argument("--num_workers", type=int, default=NUM_WORKERS)
    parser.add_argument("--pin_memory", action="store_true", help="Enable pin_memory (usually useful on CUDA, not MPS).")

    parser.add_argument("--thresh", type=float, default=THRESH, help="Threshold for binarizing sigmoid output.")
    return parser.parse_args()


# ----------------------------
# DEVICE
# ----------------------------
def get_device():
    if torch.cuda.is_available():
        return "cuda"
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return "mps"
    return "cpu"


# ----------------------------
# AUGMENTATION (TRAIN ONLY)
# ----------------------------
def get_train_transform():
    deg10 = 10.0 * np.pi / 180.0
    trans = (5, 5, 1)
    scale = (0.10, 0.10, 0.05)

    return Compose([
        RandAffined(
            keys=("image", "mask"),
            prob=0.7,
            rotate_range=(deg10, deg10, deg10),
            translate_range=trans,
            scale_range=scale,
            mode=("bilinear", "nearest"),
            padding_mode="border",
        ),
        Rand3DElasticd(
            keys=("image", "mask"),
            prob=0.3,
            sigma_range=(3.0, 5.0),
            magnitude_range=(1.0, 2.0),
            rotate_range=(0.0, 0.0, 0.0),
            translate_range=(0.0, 0.0, 0.0),
            scale_range=(0.0, 0.0, 0.0),
            mode=("bilinear", "nearest"),
            padding_mode="border",
        ),
        # intensity transforms only on image (all channels together)
        RandScaleIntensityd(keys=("image",), factors=0.10, prob=0.5),
        RandShiftIntensityd(keys=("image",), offsets=0.10, prob=0.5),
        RandGaussianNoised(keys=("image",), prob=0.3, mean=0.0, std=0.01),
    ])


# ----------------------------
# CASES
# ----------------------------
def find_cases(groups, channel_files, mask_name):
    """
    Returns list of (group, folder, [ch1_path, ch2_path, ch3_path], mask_path)
    """
    cases = []
    for group_name, root in groups.items():
        if not root or not os.path.isdir(root):
            if root != "":
                print(f'Warning: root not found or invalid for group "{group_name}": {root}')
            continue

        for dirpath, _, filenames in os.walk(root):
            ok = all(ch in filenames for ch in channel_files) and (mask_name in filenames)
            if ok:
                ch_paths = [os.path.join(dirpath, ch) for ch in channel_files]
                msk_path = os.path.join(dirpath, mask_name)
                folder = os.path.basename(dirpath)
                cases.append((group_name, folder, ch_paths, msk_path))
    return cases


def split_cases(cases, seed=42):
    rng = random.Random(seed)
    cases = cases[:]
    rng.shuffle(cases)
    n = len(cases)
    n_train = int(0.70 * n)
    n_val = int(0.15 * n)
    train = cases[:n_train]
    val = cases[n_train:n_train + n_val]
    test = cases[n_train + n_val:]
    return train, val, test


# ----------------------------
# DATASET
# ----------------------------
class NiftiSeg3ChDataset(Dataset):
    def __init__(self, items, transform=None):
        self.items = items
        self.transform = transform

    def __len__(self):
        return len(self.items)

    def __getitem__(self, idx):
        group, folder, ch_paths, msk_path = self.items[idx]

        vols = []
        for p in ch_paths:
            v = nib.load(p).get_fdata().astype(np.float32)
            vols.append(v)

        img = np.stack(vols, axis=0).astype(np.float32)  # (3, X, Y, Z)

        msk = nib.load(msk_path).get_fdata().astype(np.float32)
        msk = (msk > 0.5).astype(np.float32)
        msk = msk[None, ...]  # (1, X, Y, Z)

        img = torch.from_numpy(img)
        msk = torch.from_numpy(msk)

        sample = {"image": img, "mask": msk, "group": group, "folder": folder}
        if self.transform is not None:
            sample = self.transform(sample)
        return sample


# ----------------------------
# METRICS
# ----------------------------
def dice_score(pred, target, eps=1e-8):
    inter = (pred * target).sum(dim=(2, 3, 4))
    den = pred.sum(dim=(2, 3, 4)) + target.sum(dim=(2, 3, 4))
    d = (2 * inter + eps) / (den + eps)
    return d.mean().item()


def precision_recall(pred, target, eps=1e-8):
    tp = (pred * target).sum(dim=(2, 3, 4))
    fp = (pred * (1 - target)).sum(dim=(2, 3, 4))
    fn = ((1 - pred) * target).sum(dim=(2, 3, 4))
    precision = (tp + eps) / (tp + fp + eps)
    recall = (tp + eps) / (tp + fn + eps)
    return precision.mean().item(), recall.mean().item()


# ----------------------------
# MAIN
# ----------------------------
def main():
    args = parse_args()

    try:
        mp.set_start_method("spawn", force=True)
    except RuntimeError:
        pass

    device = get_device()
    print("Device:", device)

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)

    groups = {
        "Site1": args.Site1_root,
        "Site2": args.Site2_root,
        "Site3": args.Site3_root,
    }

    cases = find_cases(groups, args.channel_files, args.mask_name)
    if len(cases) == 0:
        raise ValueError(
            "No cases found. Check channel files + mask exist in each case folder.\n"
            f"channel_files={args.channel_files}\nmask_name={args.mask_name}"
        )

    print("Total cases found:", len(cases))
    train_items, val_items, test_items = split_cases(cases, seed=args.seed)
    print("Train/Val/Test:", len(train_items), len(val_items), len(test_items))

    train_transform = get_train_transform()

    train_ds = NiftiSeg3ChDataset(train_items, transform=train_transform)
    val_ds = NiftiSeg3ChDataset(val_items, transform=None)
    test_ds = NiftiSeg3ChDataset(test_items, transform=None)

    train_loader = DataLoader(
        train_ds, batch_size=args.batch_size, shuffle=True,
        num_workers=args.num_workers, pin_memory=args.pin_memory
    )
    val_loader = DataLoader(
        val_ds, batch_size=1, shuffle=False,
        num_workers=args.num_workers, pin_memory=args.pin_memory
    )
    test_loader = DataLoader(
        test_ds, batch_size=1, shuffle=False,
        num_workers=args.num_workers, pin_memory=args.pin_memory
    )

    model = UNet(
        spatial_dims=3,
        in_channels=3,
        out_channels=1,
        channels=(16, 32, 64, 128),
        strides=(2, 2, 1),
        num_res_units=2
    ).to(device)

    loss_fn = DiceLoss(sigmoid=True)
    opt = torch.optim.Adam(model.parameters(), lr=args.lr)
    inferer = SimpleInferer()

    # NO early stopping, but we still save best val checkpoint
    best_val_dice = -1.0

    train_losses, val_losses = [], []
    train_dices, val_dices = [], []

    for epoch in range(1, args.max_epochs + 1):
        # ---- Train ----
        model.train()
        epoch_loss = 0.0
        epoch_dice = 0.0
        n_batches = 0

        for batch in train_loader:
            img = batch["image"].to(device)
            msk = batch["mask"].to(device)

            opt.zero_grad(set_to_none=True)
            logits = model(img)
            loss = loss_fn(logits, msk)
            loss.backward()
            opt.step()

            epoch_loss += loss.item()

            with torch.no_grad():
                prob = torch.sigmoid(logits)
                pred = (prob > args.thresh).float()
                epoch_dice += dice_score(pred, msk)

            n_batches += 1

        train_loss = epoch_loss / max(1, n_batches)
        train_dice = epoch_dice / max(1, n_batches)
        train_losses.append(train_loss)
        train_dices.append(train_dice)

        # ---- Val ----
        model.eval()
        v_loss = 0.0
        v_dice = 0.0
        v_n = 0

        with torch.no_grad():
            for batch in val_loader:
                img = batch["image"].to(device)
                msk = batch["mask"].to(device)

                logits = inferer(img, model)
                loss = loss_fn(logits, msk)
                v_loss += loss.item()

                prob = torch.sigmoid(logits)
                pred = (prob > args.thresh).float()
                v_dice += dice_score(pred, msk)

                v_n += 1

        val_loss = v_loss / max(1, v_n)
        val_dice = v_dice / max(1, v_n)
        val_losses.append(val_loss)
        val_dices.append(val_dice)

        print(f"Epoch {epoch:02d} | "
              f"train loss {train_loss:.4f}, dice {train_dice:.4f} | "
              f"val loss {val_loss:.4f}, dice {val_dice:.4f}")

        # Save best model checkpoint (no early stopping)
        if epoch == 1 or val_dice > best_val_dice:
            best_val_dice = val_dice
            torch.save(model.state_dict(), args.best_model_path)

    # ---- Test (best checkpoint) ----
    model.load_state_dict(torch.load(args.best_model_path, map_location=device))
    model.eval()

    test_dice, test_prec, test_rec = [], [], []
    with torch.no_grad():
        for batch in test_loader:
            img = batch["image"].to(device)
            msk = batch["mask"].to(device)

            logits = inferer(img, model)
            prob = torch.sigmoid(logits)
            pred = (prob > args.thresh).float()

            d = dice_score(pred, msk)
            p, r = precision_recall(pred, msk)

            test_dice.append(d)
            test_prec.append(p)
            test_rec.append(r)

    print("\nTEST RESULTS (best model):")
    print("Mean Dice     :", float(np.mean(test_dice)))
    print("Mean Precision:", float(np.mean(test_prec)))
    print("Mean Recall   :", float(np.mean(test_rec)))
    print("Best val Dice :", float(best_val_dice))

    # Save final model (last epoch)
    torch.save(model.state_dict(), args.final_model_path)
    print("\nSaved models:", args.best_model_path, "and", args.final_model_path)

    # ---- Plots ----
    plt.figure()
    plt.plot(train_losses, label="train loss")
    plt.plot(val_losses, label="val loss")
    plt.xlabel("epoch")
    plt.ylabel("loss")
    plt.legend()
    plt.title("Loss curve")
    plt.tight_layout()
    plt.savefig(args.loss_plot_path, dpi=200)

    plt.figure()
    plt.plot(train_dices, label="train dice")
    plt.plot(val_dices, label="val dice")
    plt.xlabel("epoch")
    plt.ylabel("dice")
    plt.legend()
    plt.title("Dice curve")
    plt.tight_layout()
    plt.savefig(args.dice_plot_path, dpi=200)

    print("Saved plots:", args.loss_plot_path, "and", args.dice_plot_path)


if __name__ == "__main__":
    main()
