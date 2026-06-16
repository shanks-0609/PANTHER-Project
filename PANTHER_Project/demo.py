# =============================================================================
# demo.py
# PANTHER Project - Visual Demonstration
#
# Generates a side-by-side comparison of the Original MRI and the Predicted
# Tumor Mask slice overlaid on the MRI, showcasing the model's capabilities.
# Automatically picks the slice with the largest tumor.
# =============================================================================

import os
import torch
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
import torch.nn.functional as F

from monai.inferers import sliding_window_inference
from dataset import get_train_val_loaders
from model import AttentionUNet3D
import config

def get_best_slice_index(mask):
    """Find the Z-slice with the maximum tumor area."""
    # mask shape: (D, H, W)
    sums = mask.sum(axis=(1, 2))
    best_slice = np.argmax(sums)
    return best_slice

def run_demo():
    print("[Demo] Setting up...")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if not os.path.exists(config.BEST_MODEL_PATH):
        print(f"[Demo] Please train the model first. No checkpoint found at {config.BEST_MODEL_PATH}")
        return

    model = AttentionUNet3D().to(device)
    checkpoint = torch.load(config.BEST_MODEL_PATH, map_location=device)
    model.load_state_dict(checkpoint["model_state"])
    model.eval()

    print("[Demo] Model loaded successfully.")

    # Get one sample from validation set
    _, val_loader = get_train_val_loaders()
    batch = next(iter(val_loader))
    
    images = batch["image"].to(device)   # (1, 1, D, H, W)
    labels = batch["label"].to(device)   # (1, 1, D, H, W)

    print("[Demo] Running inference...")
    with torch.no_grad():
        logits = sliding_window_inference(
            inputs=images,
            roi_size=config.PATCH_SIZE,
            sw_batch_size=4,
            predictor=model,
            overlap=config.PATCH_OVERLAP,
            mode="gaussian"
        )
        probs = F.softmax(logits, dim=1)
        pred_mask = (probs[:, 1] > config.INFERENCE_THRESHOLD).float()

    # Convert to numpy arrays for plotting
    # Assuming spatial axes are D, H, W. We'll pick a slice along D.
    img_vol = images[0, 0].cpu().numpy()
    pred_vol = pred_mask[0].cpu().numpy()
    true_vol = labels[0, 0].cpu().numpy()

    best_idx = get_best_slice_index(pred_vol)
    if pred_vol.sum() == 0:
        print("[Demo] Model predicted no tumor in this sample. Proceeding with center slice.")
        best_idx = img_vol.shape[0] // 2

    print(f"[Demo] Best slice found at index Z={best_idx}")

    img_slice = img_vol[best_idx, :, :]
    mask_slice = pred_vol[best_idx, :, :]
    true_slice = true_vol[best_idx, :, :]

    # Setup the plot
    plt.style.use('dark_background')
    fig, axes = plt.subplots(1, 2, figsize=(14, 7))
    fig.patch.set_facecolor('#1e1e1e')
    
    # Original
    axes[0].imshow(img_slice, cmap='gray', origin='lower')
    axes[0].set_title('Original Contrast-Enhanced MRI', color='white', fontsize=16, pad=15)
    axes[0].axis('off')

    # Overlay
    axes[1].imshow(img_slice, cmap='gray', origin='lower')
    # Custom colormap that is completely transparent for 0, and semi-transparent red for 1
    custom_cmap = ListedColormap(['none', 'red'])
    axes[1].imshow(mask_slice, cmap=custom_cmap, alpha=0.5, origin='lower')
    axes[1].set_title('AI-Predicted Tumor Boundary', color='white', fontsize=16, pad=15)
    axes[1].axis('off')

    plt.tight_layout()
    output_path = "output_demo.png"
    plt.savefig(output_path, dpi=300, facecolor=fig.get_facecolor(), bbox_inches='tight')
    plt.close()

    print(f"[Demo] Success! Impressive outcome saved to {os.path.abspath(output_path)}")
    print(f"[Demo] Ground truth max mask value: {true_slice.max()}, Predicted max mask value: {mask_slice.max()}")

if __name__ == "__main__":
    run_demo()
