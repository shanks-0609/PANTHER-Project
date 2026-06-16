# =============================================================================
# train_task2.py
# PANTHER Task 2 – MR-Linac Adaptive Radiotherapy Segmentation
#
# Flow:
#   1. Load the best fine-tuned model from Task 1.
#   2. Train on the 50 labeled MR-Linac MRIs (Task 2 data) with differential
#      learning rates: low LR for the encoder (retaining Task 1 features) and
#      higher LR for the decoder adapting to the MR-Linac domain.
#   3. Save best Task 2 model checkpoint based on validation DSC.
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

def load_task1_best_model(model):
    """Load weights from the best Task 1 model."""
    if not os.path.exists(config.BEST_MODEL_PATH):
        print("[Task2] WARNING: No Task 1 best model found. Training from scratch.")
        return model

    checkpoint = torch.load(config.BEST_MODEL_PATH, map_location="cpu")
    model.load_state_dict(checkpoint["model_state"])
    print(f"[Task2] ✓ Loaded best Task 1 model from {config.BEST_MODEL_PATH}")
    return model

def get_post_transforms():
    post_pred  = AsDiscrete(argmax=True, to_onehot=config.OUT_CHANNELS)
    post_label = AsDiscrete(to_onehot=config.OUT_CHANNELS)
    return post_pred, post_label

def validate(model, val_loader, criterion, dice_metric, post_pred, post_label, device):
    model.eval()
    val_loss = 0.0

    with torch.no_grad():
        for batch in tqdm(val_loader, desc="  Validating", leave=False):
            images  = batch["image"].to(device)
            labels  = batch["label"].to(device)

            # Binarize labels: any value > 0 → tumor (1)
            labels = (labels > 0).long()

            logits = sliding_window_inference(
                inputs       = images,
                roi_size     = config.PATCH_SIZE,
                sw_batch_size= 6,      # Optimized for 8GB VRAM
                predictor    = model,
                overlap      = 0.25,   # Reduced from 0.5 for much faster validation
                mode         = "gaussian"
            )

            loss, _, _ = criterion(logits, labels)
            val_loss  += loss.item()

            outputs_list = decollate_batch(logits)
            labels_list  = decollate_batch(labels)
            preds_oh     = [post_pred(o) for o in outputs_list]
            labels_oh    = [post_label(l) for l in labels_list]

            dice_metric(y_pred=preds_oh, y=labels_oh)

    dice_scores = dice_metric.aggregate()         # shape: (1,) with include_background=False
    tumor_dice  = dice_scores[0].item()           # Index 0 = tumor (only foreground class)
    mean_dice   = dice_scores.mean().item()
    dice_metric.reset()

    avg_val_loss = val_loss / len(val_loader)
    return avg_val_loss, tumor_dice, mean_dice


def train_task2():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[Task2] Using device: {device}")

    # ── Data ─────────────────────────────────────────────────────────────────
    train_loader, val_loader = get_train_val_loaders(task_dir=config.TASK2_LABELED_DIR)

    # ── Model ─────────────────────────────────────────────────────────────────
    model = AttentionUNet3D().to(device)
    model = load_task1_best_model(model)

    # ── Differential Learning Rates ──────────────────────────────────────────
    encoder_params = []
    decoder_params = []
    for name, param in model.named_parameters():
        if "enc" in name or "bottleneck" in name:
            encoder_params.append(param)
        else:
            decoder_params.append(param)

    optimizer = AdamW([
        {"params": encoder_params, "lr": config.TASK2_LR_ENCODER},
        {"params": decoder_params, "lr": config.TASK2_LR_DECODER}
    ], weight_decay=config.WEIGHT_DECAY)

    # ── Loss, optimizer, scheduler ────────────────────────────────────────────
    criterion = CombinedDiceBCELoss()
    warmup_scheduler  = LinearLR(optimizer, start_factor=0.1, end_factor=1.0, total_iters=config.WARMUP_EPOCHS)
    cosine_scheduler  = CosineAnnealingLR(optimizer, T_max=config.TASK2_EPOCHS - config.WARMUP_EPOCHS, eta_min=1e-6)
    scheduler         = SequentialLR(optimizer, schedulers=[warmup_scheduler, cosine_scheduler], milestones=[config.WARMUP_EPOCHS])

    # ── Metrics ───────────────────────────────────────────────────────────────
    dice_metric            = DiceMetric(include_background=False, reduction="mean_batch")
    post_pred, post_label  = get_post_transforms()

    # ── Logging ───────────────────────────────────────────────────────────────
    writer = SummaryWriter(log_dir=os.path.join(config.LOG_DIR, "task2"))
    scaler = torch.amp.GradScaler('cuda')

    # ── Training state ────────────────────────────────────────────────────────
    best_val_dice     = -1.0
    no_improve_count  = 0
    start_time        = time.time()

    print(f"\n[Task2] Starting transfer learning for {config.TASK2_EPOCHS} epochs...\n")

    for epoch in range(1, config.TASK2_EPOCHS + 1):
        model.train()
        train_loss = 0.0
        train_dice = 0.0
        n_steps    = 0

        pbar = tqdm(train_loader, desc=f"Epoch {epoch:3d}/{config.TASK2_EPOCHS}")

        for batch in pbar:
            images = batch["image"].to(device)
            labels = batch["label"].to(device)

            optimizer.zero_grad()
            
            with torch.amp.autocast('cuda'):
                logits = model(images)
                loss, d_loss, c_loss = criterion(logits, labels)
            
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), config.GRAD_CLIP)
            
            scaler.step(optimizer)
            scaler.update()

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
                "dice": f"{batch_dice.item():.4f}"
            })

        scheduler.step()
        current_lr = scheduler.get_last_lr()[-1] # take decoder LR to display

        avg_train_loss = train_loss / n_steps
        avg_train_dice = train_dice / n_steps

        writer.add_scalar("Loss/train_task2",       avg_train_loss, epoch)
        writer.add_scalar("Dice/train_task2",       avg_train_dice, epoch)
        writer.add_scalar("LearningRate_Task2",     current_lr,     epoch)

        print(f"[Epoch {epoch:3d}] Train Loss: {avg_train_loss:.4f} | "
              f"Train Dice: {avg_train_dice:.4f} | LR(Dec): {current_lr:.2e}")

        # Validation
        if epoch % config.VAL_INTERVAL == 0:
            val_loss, tumor_dice, mean_dice = validate(
                model, val_loader, criterion, dice_metric,
                post_pred, post_label, device
            )

            writer.add_scalar("Loss/val_task2",       val_loss,   epoch)
            writer.add_scalar("Dice/val_tumor_task2", tumor_dice, epoch)

            elapsed = (time.time() - start_time) / 60
            print(f"  ┌─ Validation Task 2 ──────────────────────────────────")
            print(f"  │  Val Loss   : {val_loss:.4f}")
            print(f"  │  Tumor Dice : {tumor_dice:.4f}")
            print(f"  │  Time so far: {elapsed:.1f} min")
            print(f"  └──────────────────────────────────────────────────────")

            if tumor_dice > best_val_dice:
                best_val_dice    = tumor_dice
                no_improve_count = 0
                torch.save({
                    "epoch":           epoch,
                    "model_state":     model.state_dict(),
                    "optimizer_state": optimizer.state_dict(),
                    "best_val_dice":   best_val_dice,
                }, config.TASK2_BEST_MODEL)
                print(f"  ★ New best model! Tumor Dice: {best_val_dice:.4f} → {config.TASK2_BEST_MODEL}")
            else:
                no_improve_count += 1
                print(f"  (No improvement for {no_improve_count * config.VAL_INTERVAL} epochs. "
                      f"Best: {best_val_dice:.4f})")

            if no_improve_count >= config.EARLY_STOP_PATIENCE:
                print(f"\n[Task2] Early stopping at epoch {epoch} — "
                      f"no improvement for {config.EARLY_STOP_PATIENCE} validation rounds.")
                break

    total_time = (time.time() - start_time) / 60
    writer.close()
    print(f"\n[Task2] Transfer learning complete!")
    print(f"[Task2] Best validation Tumor Dice : {best_val_dice:.4f}")
    print(f"[Task2] Total training time        : {total_time:.1f} minutes")
    print(f"[Task2] Best model saved to        : {config.TASK2_BEST_MODEL}")


if __name__ == "__main__":
    train_task2()
