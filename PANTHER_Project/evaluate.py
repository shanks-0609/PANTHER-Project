# =============================================================================
# evaluate.py
# PANTHER Task 1 – Evaluation metrics and inference on validation set
#
# Metrics computed (consistent with PANTHER challenge evaluation):
#   1. Dice Similarity Coefficient (DSC)  — primary metric
#   2. 95th Percentile Hausdorff Distance (HD95)
#   3. Normalized Surface Dice (NSD)      — at 2 mm tolerance
#   4. Average Symmetric Surface Distance (ASSD)
# =============================================================================

import os
import numpy as np
import torch
import pandas as pd
import matplotlib.pyplot as plt
from tqdm import tqdm
from monai.inferers import sliding_window_inference
from monai.metrics import (
    DiceMetric,
    HausdorffDistanceMetric,
    SurfaceDiceMetric,
    SurfaceDistanceMetric,
)
from monai.transforms import AsDiscrete
from monai.data import decollate_batch

import config
from dataset import get_train_val_loaders
from model import AttentionUNet3D


# ─── Load model ───────────────────────────────────────────────────────────────

def load_model(checkpoint_path, device):
    """Load best model from checkpoint."""
    model      = AttentionUNet3D().to(device)
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint["model_state"])
    model.eval()

    epoch     = checkpoint.get("epoch", "?")
    best_dice = checkpoint.get("best_val_dice", "?")
    print(f"[Evaluate] Loaded model from epoch {epoch} | "
          f"Saved Val Dice: {best_dice:.4f}" if isinstance(best_dice, float)
          else f"[Evaluate] Loaded checkpoint from {checkpoint_path}")
    return model


# ─── Per-sample evaluation ────────────────────────────────────────────────────

def evaluate():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[Evaluate] Device: {device}")

    if not os.path.exists(config.BEST_MODEL_PATH):
        raise FileNotFoundError(f"No checkpoint found at {config.BEST_MODEL_PATH}. "
                                f"Run train.py first.")

    _, val_loader = get_train_val_loaders()
    model         = load_model(config.BEST_MODEL_PATH, device)

    # ── Metrics (MONAI) ───────────────────────────────────────────────────────
    dice_metric  = DiceMetric(include_background=False, reduction="none")
    hd95_metric  = HausdorffDistanceMetric(include_background=False, percentile=95, reduction="none")
    nsd_metric   = SurfaceDiceMetric(include_background=False, class_thresholds=[2.0], reduction="none")
    assd_metric  = SurfaceDistanceMetric(include_background=False, symmetric=True, reduction="none")

    post_pred  = AsDiscrete(argmax=True, to_onehot=config.OUT_CHANNELS)
    post_label = AsDiscrete(to_onehot=config.OUT_CHANNELS)

    results = []

    print(f"\n[Evaluate] Running inference on {len(val_loader)} validation volumes...\n")

    with torch.no_grad():
        for idx, batch in enumerate(tqdm(val_loader, desc="Evaluating")):
            images = batch["image"].to(device)
            labels = batch["label"].to(device)

            # Binarize labels: any value > 0 -> tumor (1)
            labels = (labels > 0).long()

            fname  = batch.get("image_meta_dict", {}).get("filename_or_obj", [f"case_{idx:03d}"])[0]

            # Sliding window inference (handles full 3D volume)
            logits = sliding_window_inference(
                inputs        = images,
                roi_size      = config.PATCH_SIZE,
                sw_batch_size = 4,
                predictor     = model,
                overlap       = config.PATCH_OVERLAP,
                mode          = "gaussian"
            )

            # Post-process
            outputs_list = decollate_batch(logits)
            labels_list  = decollate_batch(labels)
            preds_oh     = [post_pred(o) for o in outputs_list]
            labels_oh    = [post_label(l) for l in labels_list]

            # Compute metrics
            dice_metric(y_pred=preds_oh,  y=labels_oh)
            hd95_metric(y_pred=preds_oh,  y=labels_oh)
            nsd_metric(y_pred=preds_oh,   y=labels_oh)
            assd_metric(y_pred=preds_oh,  y=labels_oh)

    # ── Aggregate results ─────────────────────────────────────────────────────
    all_dice  = dice_metric.aggregate().cpu().numpy()    # (N, C-1)
    all_hd95  = hd95_metric.aggregate().cpu().numpy()
    all_nsd   = nsd_metric.aggregate().cpu().numpy()
    all_assd  = assd_metric.aggregate().cpu().numpy()

    # Extract tumor class (index 0 after removing background)
    tumor_dice = all_dice[:, 0]
    tumor_hd95 = all_hd95[:, 0]
    tumor_nsd  = all_nsd[:, 0]
    tumor_assd = all_assd[:, 0]

    # ── Summary statistics ────────────────────────────────────────────────────
    print("\n" + "="*60)
    print("  PANTHER Task 1 – Evaluation Results")
    print("="*60)
    print(f"  Metric                     Mean ± Std")
    print("-"*60)
    print(f"  Dice (DSC)          :  {tumor_dice.mean():.4f} ± {tumor_dice.std():.4f}")
    print(f"  HD95 (mm)           :  {tumor_hd95.mean():.2f}  ± {tumor_hd95.std():.2f}")
    print(f"  NSD (2mm tolerance) :  {tumor_nsd.mean():.4f} ± {tumor_nsd.std():.4f}")
    print(f"  ASSD (mm)           :  {tumor_assd.mean():.2f}  ± {tumor_assd.std():.2f}")
    print("="*60)

    # ── Save per-sample CSV ───────────────────────────────────────────────────
    csv_path = os.path.join(config.OUTPUT_DIR, "evaluation_results.csv")
    df = pd.DataFrame({
        "Case":      [f"case_{i:03d}" for i in range(len(tumor_dice))],
        "DSC":       tumor_dice,
        "HD95_mm":   tumor_hd95,
        "NSD_2mm":   tumor_nsd,
        "ASSD_mm":   tumor_assd,
    })
    df.loc["Mean"] = ["MEAN", tumor_dice.mean(), tumor_hd95.mean(), tumor_nsd.mean(), tumor_assd.mean()]
    df.to_csv(csv_path, index=False)
    print(f"\n[Evaluate] Per-sample results saved to: {csv_path}")

    # ── Plot Dice distribution ────────────────────────────────────────────────
    plot_path = os.path.join(config.OUTPUT_DIR, "dice_distribution.png")
    plt.figure(figsize=(8, 5))
    plt.hist(tumor_dice[:-1], bins=15, color="#2E75B6", edgecolor="white", alpha=0.8)
    plt.axvline(tumor_dice[:-1].mean(), color="red", linestyle="--",
                label=f"Mean DSC = {tumor_dice[:-1].mean():.3f}")
    plt.xlabel("Dice Similarity Coefficient", fontsize=12)
    plt.ylabel("Number of Cases",             fontsize=12)
    plt.title("Task 1 – Tumor DSC Distribution (Validation Set)", fontsize=13, fontweight="bold")
    plt.legend(fontsize=11)
    plt.tight_layout()
    plt.savefig(plot_path, dpi=150)
    plt.close()
    print(f"[Evaluate] DSC distribution plot saved to: {plot_path}")

    return {
        "mean_dice": tumor_dice[:-1].mean(),
        "mean_hd95": tumor_hd95[:-1].mean(),
        "mean_nsd":  tumor_nsd[:-1].mean(),
        "mean_assd": tumor_assd[:-1].mean(),
    }


# ─── Single volume inference ──────────────────────────────────────────────────

def predict_volume(image_path, model=None, device=None, save_path=None):
    """
    Run inference on a single NIfTI volume and optionally save the prediction.

    Args:
        image_path (str): Path to a .nii.gz MRI file
        model      : Loaded AttentionUNet3D model (loaded from checkpoint if None)
        device     : torch.device
        save_path  (str): If provided, saves the prediction mask as .nii.gz

    Returns:
        pred_mask (np.ndarray): Binary prediction mask (same shape as input)
    """
    import nibabel as nib
    from monai.transforms import Compose, LoadImage, EnsureChannelFirst, \
        Spacing, Orientation, NormalizeIntensity, CropForeground, EnsureType

    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if model is None:
        model = load_model(config.BEST_MODEL_PATH, device)

    # Minimal preprocessing
    transforms = Compose([
        LoadImage(image_only=True),
        EnsureChannelFirst(),
        EnsureType(),
        Spacing(pixdim=config.TARGET_SPACING, mode="bilinear"),
        Orientation(axcodes="RAS"),
        NormalizeIntensity(nonzero=True, channel_wise=True),
        CropForeground(source_key=None),
    ])

    processed = transforms(image_path).unsqueeze(0).to(device)  # Add batch dim

    with torch.no_grad():
        logits = sliding_window_inference(
            inputs        = processed,
            roi_size      = config.PATCH_SIZE,
            sw_batch_size = 4,
            predictor     = model,
            overlap       = config.PATCH_OVERLAP,
            mode          = "gaussian"
        )

    probs     = torch.softmax(logits, dim=1)
    pred_mask = (probs[0, 1] > config.INFERENCE_THRESHOLD).cpu().numpy().astype(np.uint8)

    if save_path:
        nib.save(nib.Nifti1Image(pred_mask, affine=np.eye(4)), save_path)
        print(f"[Predict] Saved prediction to: {save_path}")

    return pred_mask


if __name__ == "__main__":
    evaluate()
