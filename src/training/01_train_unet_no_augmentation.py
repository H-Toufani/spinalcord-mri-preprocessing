"""

3D U-Net (MONAI) on ALL data with 70/15/15 split
NO augmentation

- Finds all cases across 3 groups
- Shuffles + splits: 70% train, 15% val, 15% test
- Trains up to 50 epochs with early stopping (best by val Dice)
- Saves best + final model
- Evaluates on test: Dice, Precision, Recall
- Plots: loss curves + Dice curves (train/val)

IMPORTANT:
- IMAGE_NAME and MASK_NAME must exist in each case folder
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


# ----------------------------
# SETTINGS
# ----------------------------
# Default to env vars, but can also be passed via CLI arguments.
GROUPS = {
    "Site1": os.environ.get("Site1_ROOT", ""),
    "Site2": os.environ.get("Site2_ROOT", ""),
    "Site3": os.environ.get("Site3_ROOT", ""),
}

# Use your denoised cropped images (change if needed)
IMAGE_NAME = "image.nii.gz"  # input image filename
# IMPORTANT: mask must be in the SAME space and SAME crop size as image
MASK_NAME = "T2s_seg_cropped.nii.gz"  # mask filename

BEST_MODEL_PATH = "best_model_unet.pth"
FINAL_MODEL_PATH = "final_model_unet.pth"
LOSS_PLOT_PATH = "loss_curve.png"
DICE_PLOT_PATH = "dice_curve.png"

SEED = 42
MAX_EPOCHS = 50
EARLY_STOP_PATIENCE = 8
LR = 1e-3

BATCH_SIZE = 4
NUM_WORKERS = 2
PIN_MEMORY = False  # MPS doesn't benefit

THRESH = 0.5


def parse_args():
    parser = argparse.ArgumentParser(description="3D U-Net (MONAI) training with 70/15/15 split, no augmentation.")
    parser.add_argument("--site1_root", type=str, default=GROUPS["Site1"], help="Root directory for Site1.")
    parser.add_argument("--site2_root", type=str, default=GROUPS["Site2"], help="Root directory for Site2.")
    parser.add_argument("--site3_root",  type=str, default=GROUPS["Site3"], help="Root directory for Site3.")

    parser.add_argument("--image_name", type=str, default=IMAGE_NAME, help="Input image filename in each case folder.")
    parser.add_argument("--mask_name", type=str, default=MASK_NAME, help="Mask filename in each case folder.")

    parser.add_argument("--best_model_path", type=str, default=BEST_MODEL_PATH)
    parser.add_argument("--final_model_path", type=str, default=FINAL_MODEL_PATH)
    parser.add_argument("--loss_plot_path", type=str, default=LOSS_PLOT_PATH)
    parser.add_argument("--dice_plot_path", type=str, default=DICE_PLOT_PATH)

    parser.add_argument("--seed", type=int, default=SEED)
    parser.add_argument("--max_epochs", type=int, default=MAX_EPOCHS)
    parser.add_argument("--early_stop_patience", type=int, default=EARLY_STOP_PATIENCE)
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
# CASES
# ----------------------------
def find_cases(groups, image_name, mask_name):
    cases = []
    for group_name, root in groups.items():
        if not root or not os.path.isdir(root):
            if root != "":
                print(f'Warning: root not found or invalid for group "{group_name}": {root}')
            continue

        for dirpath, _, filenames in os.walk(root):
            if image_name in filenames and mask_name in filenames:
                img_path = os.path.join(dirpath, image_name)
                msk_path = os.path.join(dirpath, mask_name)
                folder = os.path.basename(dirpath)
                cases.append((group_name, folder, img_path, msk_path))
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
class NiftiSegDataset(Dataset):
    def __init__(self, items):
        self.items = items

    def __len__(self):
        return len(self.items)

    def __getitem__(self, idx):
        group, folder, img_path, msk_path = self.items[idx]

        img = nib.load(img_path).get_fdata().astype(np.float32)  # (X,Y,Z)
        msk = nib.load(msk_path).get_fdata().astype(np.float32)  # (X,Y,Z)
        msk = (msk > 0.5).astype(np.float32)

        img = torch.from_numpy(img[None, ...])  # (1,X,Y,Z)
        msk = torch.from_numpy(msk[None, ...])  # (1,X,Y,Z)

        return {"image": img, "mask": msk, "group": group, "folder": folder}


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
        "Site1": args.site1_root,
        "Site2": args.site2_root,
        "Site3": args.site3_root,
    }

    cases = find_cases(groups, args.image_name, args.mask_name)
    if len(cases) == 0:
        raise ValueError("No cases found. Check dataset roots and IMAGE_NAME/MASK_NAME.")

    print("Total cases found:", len(cases))

    train_items, val_items, test_items = split_cases(cases, seed=args.seed)
    print("Train/Val/Test:", len(train_items), len(val_items), len(test_items))

    train_ds = NiftiSegDataset(train_items)
    val_ds = NiftiSegDataset(val_items)
    test_ds = NiftiSegDataset(test_items)

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
        in_channels=1,
        out_channels=1,
        channels=(16, 32, 64, 128),
        strides=(2, 2, 1),
        num_res_units=2
    ).to(device)

    loss_fn = DiceLoss(sigmoid=True)
    opt = torch.optim.Adam(model.parameters(), lr=args.lr)
    inferer = SimpleInferer()

    best_val_dice = -1.0
    best_epoch = -1
    patience = 0

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

        # ---- Validation ----
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

        # ---- Early stopping ----
        if val_dice > best_val_dice + 1e-4:
            best_val_dice = val_dice
            best_epoch = epoch
            patience = 0
            torch.save(model.state_dict(), args.best_model_path)
        else:
            patience += 1
            if patience >= args.early_stop_patience:
                print(f"Early stopping at epoch {epoch}. Best epoch={best_epoch}, best val dice={best_val_dice:.4f}")
                break

    # Load best model and test
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
