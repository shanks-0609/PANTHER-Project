# =============================================================================
# losses.py
# PANTHER Task 1 – Combined Dice + Binary Cross-Entropy loss
# =============================================================================

import torch
import torch.nn as nn
import torch.nn.functional as F
import config


class DiceLoss(nn.Module):
    """
    Soft Dice loss for binary or multi-class segmentation.
    Operates on softmax probabilities (not logits).
    Averages across all foreground classes (excludes background class 0).
    """

    def __init__(self, smooth=1e-5):
        super().__init__()
        self.smooth = smooth

    def forward(self, probs, targets):
        """
        Args:
            probs   : (B, C, D, H, W) — softmax probabilities
            targets : (B, 1, D, H, W) — integer class labels
        """
        num_classes = probs.shape[1]

        # Binarize: any label > 0 is treated as tumor (handles multi-class .mha files)
        targets_bin = (targets > 0).long()

        # One-hot encode targets → (B, C, D, H, W)
        targets_onehot = F.one_hot(
            targets_bin.squeeze(1), num_classes=num_classes
        ).permute(0, 4, 1, 2, 3).float()

        # Compute Dice per class, skip background (class 0)
        dice_scores = []
        for c in range(1, num_classes):
            p = probs[:, c]
            g = targets_onehot[:, c]
            intersection = (p * g).sum()
            dice = (2.0 * intersection + self.smooth) / (p.sum() + g.sum() + self.smooth)
            dice_scores.append(dice)

        mean_dice = torch.stack(dice_scores).mean()
        return 1.0 - mean_dice   # Loss = 1 - Dice


class CombinedDiceBCELoss(nn.Module):
    """
    Combined loss: λ_dice × DiceLoss + λ_bce × CrossEntropyLoss
    
    - Dice loss  : handles class imbalance, penalizes missed tumor voxels
    - BCE (CE)   : provides stable gradients throughout training
    """

    def __init__(
        self,
        dice_weight = config.DICE_WEIGHT,
        bce_weight  = config.BCE_WEIGHT
    ):
        super().__init__()
        self.dice_weight = dice_weight
        self.bce_weight  = bce_weight
        self.dice_loss   = DiceLoss()
        self.ce_loss     = nn.CrossEntropyLoss()

    def forward(self, logits, targets):
        """
        Args:
            logits  : (B, C, D, H, W) — raw model outputs (before softmax)
            targets : (B, 1, D, H, W) — integer class labels (0=bg, 1=tumor)
        Returns:
            total_loss, dice_loss, ce_loss
        """
        # Binarize: treat any label > 0 as tumor (handles multi-class .mha labels)
        targets = (targets > 0).long()

        probs = F.softmax(logits, dim=1)

        # Dice loss on softmax probs
        d_loss = self.dice_loss(probs, targets)

        # Cross-entropy expects (B, C, D, H, W) logits and (B, D, H, W) targets
        c_loss = self.ce_loss(logits, targets.squeeze(1).long())

        total = self.dice_weight * d_loss + self.bce_weight * c_loss

        return total, d_loss, c_loss


class ReconstructionLoss(nn.Module):
    """
    MSE reconstruction loss for self-supervised pretraining.
    Used to train the encoder to reconstruct corrupted MRI volumes.
    """

    def __init__(self):
        super().__init__()
        self.mse = nn.MSELoss()

    def forward(self, reconstruction, original):
        return self.mse(reconstruction, original)


# ─── Quick test ───────────────────────────────────────────────────────────────

if __name__ == "__main__":
    B, C, D, H, W = 2, 2, 64, 64, 32
    logits  = torch.randn(B, C, D, H, W)
    targets = torch.randint(0, C, (B, 1, D, H, W))

    criterion = CombinedDiceBCELoss()
    loss, dl, cl = criterion(logits, targets)

    print(f"Total loss : {loss.item():.4f}")
    print(f"Dice  loss : {dl.item():.4f}")
    print(f"CE    loss : {cl.item():.4f}")
