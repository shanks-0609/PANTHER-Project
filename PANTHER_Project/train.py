# =============================================================================
# train.py
# PANTHER Task 1 – Supervised fine-tuning of 3D Attention U-Net
#
# Flow:
#   1. Load pretrained encoder weights (if available from pretrain.py)
#   2. Train on 92 labeled T1-weighted MRIs (80/20 train/val split)
#   3. Validate every VAL_INTERVAL epochs using Dice Similarity Coefficient
#   4. Save best model checkpoint based on validation DSC
#   5. Log metrics to TensorBoard
# =============================================================================

import os
import time
import torch
import torch.nn.functional as F
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR, LinearLR, SequentialLR
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
from monai.metrics import DiceMetric
from monai.inferers import sliding_window_inference
from monai.transforms import AsDiscrete
from monai.data import decollate_batch

import config
from dataset import get_train_val_loaders
from model import AttentionUNet3D
from losses import CombinedDiceBCELoss


# ─── Load pretrained encoder weights ─────────────────────────────────────────

def load_pretrained_encoder(model):
    """Load encoder weights from self-supervised pretraining into the model."""
    if not os.path.exists(config.PRETRAIN_CKPT):
        print("[Train] No pretrained encoder found — training from scratch.")
        return model

    pretrained_state = torch.load(config.PRETRAIN_CKPT, map_location="cpu")
    model_state      = model.state_dict()

    # Only load keys that exist in both and have matching shapes
    loaded, skipped = 0, 0
    for k, v in pretrained_state.items():
        if k in model_state and model_state[k].shape == v.shape:
            model_state[k] = v
            loaded += 1
        else:
            skipped += 1

    model.load_state_dict(model_state)
    print(f"[Train] Loaded pretrained encoder: {loaded} layers matched, {skipped} skipped.")
    return model


# ─── Metric helpers ───────────────────────────────────────────────────────────

def get_post_transforms():
    """Post-processing: argmax + one-hot for metric computation."""
    post_pred  = AsDiscrete(argmax=True, to_onehot=config.OUT_CHANNELS)
    post_label = AsDiscrete(to_onehot=config.OUT_CHANNELS)
    return post_pred, post_label


# ─── Validation step ──────────────────────────────────────────────────────────

def validate(model, val_loader, criterion, dice_metric, post_pred, post_label, device):
    """
    Run sliding-window inference on full volumes and compute mean Dice score.
    Sliding window is used because full volumes exceed GPU memory at training resolution.
    """
    model.eval()
    val_loss = 0.0

    with torch.no_grad():
        for batch in tqdm(val_loader, desc="  Validating", leave=False):
            images  = batch["image"].to(device)
            labels  = batch["label"].to(device)

            # Binarize labels: any value > 0 → tumor (1)
            labels = (labels > 0).long()

            # Sliding window inference: run model on overlapping patches across full volume
            logits = sliding_window_inference(
                inputs       = images,
                roi_size     = config.PATCH_SIZE,
                sw_batch_size= 6,      # Safe for 8GB VRAM
                predictor    = model,
                overlap      = 0.25,   # Reduced from 0.5 → ~3x fewer patches, much faster
                mode         = "gaussian"
            )

            loss, _, _ = criterion(logits, labels)
            val_loss  += loss.item()

            # Convert logits → one-hot predictions for metric computation
            outputs_list = decollate_batch(logits)
            labels_list  = decollate_batch(labels)
            preds_oh     = [post_pred(o) for o in outputs_list]
            labels_oh    = [post_label(l) for l in labels_list]

            dice_metric(y_pred=preds_oh, y=labels_oh)

    # Aggregate Dice across all validation volumes
    # include_background=False → returns (num_foreground_classes,) = (1,) for binary
    dice_scores = dice_metric.aggregate()         # shape: (1,)
    tumor_dice  = dice_scores[0].item()           # Index 0 = tumor (only foreground class)
    mean_dice   = dice_scores.mean().item()       # Mean over all foreground classes
    dice_metric.reset()

    avg_val_loss = val_loss / len(val_loader)
    return avg_val_loss, tumor_dice, mean_dice


# ─── Main training loop ───────────────────────────────────────────────────────

def train():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[Train] Using device: {device}")
    
    if device.type == "cuda":
        torch.backends.cudnn.benchmark = True   # Faster convolutions on RTX

    # ── Data ─────────────────────────────────────────────────────────────────
    train_loader, val_loader = get_train_val_loaders()

    # ── Model ─────────────────────────────────────────────────────────────────
    model = AttentionUNet3D().to(device)
    model = load_pretrained_encoder(model)
    # Note: torch.compile requires Triton (not available on Windows) — skipped

    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"[Train] Trainable parameters: {total_params:,}")

    # ── Loss, optimizer, scheduler ────────────────────────────────────────────
    criterion = CombinedDiceBCELoss()
    optimizer = AdamW(
        model.parameters(),
        lr=config.LEARNING_RATE,
        weight_decay=config.WEIGHT_DECAY
    )

    # Warmup for first WARMUP_EPOCHS, then cosine annealing
    warmup_scheduler  = LinearLR(optimizer, start_factor=0.1, end_factor=1.0, total_iters=config.WARMUP_EPOCHS)
    cosine_scheduler  = CosineAnnealingLR(optimizer, T_max=config.EPOCHS - config.WARMUP_EPOCHS, eta_min=1e-6)
    scheduler         = SequentialLR(optimizer, schedulers=[warmup_scheduler, cosine_scheduler], milestones=[config.WARMUP_EPOCHS])

    # ── Metrics ───────────────────────────────────────────────────────────────
    dice_metric            = DiceMetric(include_background=False, reduction="mean_batch")
    post_pred, post_label  = get_post_transforms()

    # ── Logging ───────────────────────────────────────────────────────────────
    writer = SummaryWriter(log_dir=config.LOG_DIR)

    # ── AMP Scaler ────────────────────────────────────────────────────────────
    scaler = torch.amp.GradScaler('cuda')

    # ── Training state ────────────────────────────────────────────────────────
    best_val_dice     = -1.0
    no_improve_count  = 0
    start_time        = time.time()

    print(f"\n[Train] Starting training for {config.EPOCHS} epochs...\n")

    for epoch in range(1, config.EPOCHS + 1):
        model.train()
        train_loss = 0.0
        train_dice = 0.0
        n_steps    = 0

        pbar = tqdm(train_loader, desc=f"Epoch {epoch:3d}/{config.EPOCHS}")

        for batch in pbar:
            images = batch["image"].to(device)   # (B, 1, D, H, W)
            labels = batch["label"].to(device)   # (B, 1, D, H, W)

            optimizer.zero_grad()
            
            with torch.amp.autocast('cuda'):
                logits = model(images)               # (B, 2, D, H, W)
                loss, d_loss, c_loss = criterion(logits, labels)
            
            scaler.scale(loss).backward()

            # Gradient clipping for training stability
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), config.GRAD_CLIP)
            
            scaler.step(optimizer)
            scaler.update()

            # Quick per-batch Dice estimate (on training patches)
            with torch.no_grad():
                probs      = F.softmax(logits, dim=1)
                pred_mask  = (probs[:, 1] > config.INFERENCE_THRESHOLD).float()
                true_mask  = (labels[:, 0] > 0).float()
                inter      = (pred_mask * true_mask).sum()
                batch_dice = (2.0 * inter) / (pred_mask.sum() + true_mask.sum() + 1e-5)
                train_dice += batch_dice.item()

            train_loss += loss.item()
            n_steps    += 1

            pbar.set_postfix({
                "loss": f"{loss.item():.4f}",
                "dice": f"{batch_dice.item():.4f}",
                "d_loss": f"{d_loss.item():.4f}",
                "ce_loss": f"{c_loss.item():.4f}"
            })

        # Step learning rate scheduler
        scheduler.step()
        current_lr = scheduler.get_last_lr()[0]

        avg_train_loss = train_loss / n_steps
        avg_train_dice = train_dice / n_steps

        # ── Logging ─────────────────────────────────────────────────────────
        writer.add_scalar("Loss/train",       avg_train_loss, epoch)
        writer.add_scalar("Dice/train",       avg_train_dice, epoch)
        writer.add_scalar("LearningRate",     current_lr,     epoch)

        print(f"[Epoch {epoch:3d}] Train Loss: {avg_train_loss:.4f} | "
              f"Train Dice: {avg_train_dice:.4f} | LR: {current_lr:.2e}")

        # ── Validation ──────────────────────────────────────────────────────
        if epoch % config.VAL_INTERVAL == 0:
            val_loss, tumor_dice, mean_dice = validate(
                model, val_loader, criterion, dice_metric,
                post_pred, post_label, device
            )

            writer.add_scalar("Loss/val",       val_loss,   epoch)
            writer.add_scalar("Dice/val_tumor", tumor_dice, epoch)
            writer.add_scalar("Dice/val_mean",  mean_dice,  epoch)

            elapsed = (time.time() - start_time) / 60
            print(f"  ┌─ Validation ─────────────────────────────────────────")
            print(f"  │  Val Loss   : {val_loss:.4f}")
            print(f"  │  Tumor Dice : {tumor_dice:.4f}")
            print(f"  │  Mean Dice  : {mean_dice:.4f}")
            print(f"  │  Time so far: {elapsed:.1f} min")
            print(f"  └──────────────────────────────────────────────────────")

            # ── Save best model ──────────────────────────────────────────────
            if tumor_dice > best_val_dice:
                best_val_dice    = tumor_dice
                no_improve_count = 0
                torch.save({
                    "epoch":           epoch,
                    "model_state":     model.state_dict(),
                    "optimizer_state": optimizer.state_dict(),
                    "best_val_dice":   best_val_dice,
                }, config.BEST_MODEL_PATH)
                print(f"  ★ New best model! Tumor Dice: {best_val_dice:.4f} → {config.BEST_MODEL_PATH}")
            else:
                no_improve_count += 1
                print(f"  (No improvement for {no_improve_count * config.VAL_INTERVAL} epochs. "
                      f"Best: {best_val_dice:.4f})")

            # ── Early stopping ───────────────────────────────────────────────
            if no_improve_count >= config.EARLY_STOP_PATIENCE:
                print(f"\n[Train] Early stopping at epoch {epoch} — "
                      f"no improvement for {config.EARLY_STOP_PATIENCE} validation rounds.")
                break

    # ── Final summary ────────────────────────────────────────────────────────
    total_time = (time.time() - start_time) / 60
    writer.close()
    print(f"\n[Train] Training complete!")
    print(f"[Train] Best validation Tumor Dice : {best_val_dice:.4f}")
    print(f"[Train] Total training time        : {total_time:.1f} minutes")
    print(f"[Train] Best model saved to        : {config.BEST_MODEL_PATH}")


if __name__ == "__main__":
    train()
