# =============================================================================
# config.py
# PANTHER Task 1 – All hyperparameters and path settings in one place
# =============================================================================

import os

# ─── Paths ────────────────────────────────────────────────────────────────────
DATA_DIR            = "D:/Bennett/Projects/Pancreatic Tumor/dataset/Task1"
LABELED_DIR         = "D:/Bennett/Projects/Pancreatic Tumor/dataset/Task1"
UNLABELED_DIR       = "D:/Bennett/Projects/Pancreatic Tumor/dataset/Task1/ImagesTr_unlabeled"

TASK2_LABELED_DIR   = "D:/Bennett/Projects/Pancreatic Tumor/dataset/Task2"
TASK2_BEST_MODEL    = "./outputs/checkpoints/best_task2_model.pth"
OUTPUT_DIR          = "./outputs"
CHECKPOINT_DIR      = os.path.join(OUTPUT_DIR, "checkpoints")
PRETRAIN_CKPT       = os.path.join(CHECKPOINT_DIR, "pretrained_encoder.pth")
BEST_MODEL_PATH     = os.path.join(CHECKPOINT_DIR, "best_model.pth")
LOG_DIR             = os.path.join(OUTPUT_DIR, "logs")

# Each labeled sample should have:
#   LABELED_DIR/images/case_001.nii.gz
#   LABELED_DIR/masks/case_001.nii.gz

# ─── Data splits ──────────────────────────────────────────────────────────────
TRAIN_RATIO         = 0.80    # 80% of 92 = ~74 training samples
VAL_RATIO           = 0.20    # 20% of 92 = ~18 validation samples
RANDOM_SEED         = 42

# ─── Preprocessing ────────────────────────────────────────────────────────────
TARGET_SPACING      = (1.0, 1.0, 1.5)   # mm — resample all volumes to this
PATCH_SIZE          = (96, 96, 48)    # 3D input patch (x, y, z)
PATCH_OVERLAP       = 0.50              # 50% overlap during sliding window inference
CACHE_RATE          = 1.0               # Cache entire dataset in RAM (set <1 if low RAM)
NUM_WORKERS         = 0                 # SAFE MODE: Set to 0 for Windows to avoid 'shared file mapping' errors

# ─── Model ────────────────────────────────────────────────────────────────────
IN_CHANNELS         = 1        # Grayscale MRI
OUT_CHANNELS        = 2        # Background + Tumor
FEATURE_CHANNELS    = (32, 64, 128, 256, 512)   # Encoder feature sizes per level
DROPOUT_RATE        = 0.2

# ─── Loss ─────────────────────────────────────────────────────────────────────
DICE_WEIGHT         = 0.75
BCE_WEIGHT          = 0.25

# ─── Training ─────────────────────────────────────────────────────────────────
EPOCHS              = 300        # Extended for better convergence
TASK2_EPOCHS        = 150        # Extended for better convergence
BATCH_SIZE          = 2
LEARNING_RATE       = 3e-4       # Slightly increased for better optimisation with AdamW
TASK2_LR_ENCODER    = 1e-5       # Lower LR for pretrained encoder
TASK2_LR_DECODER    = 3e-4       # Higher LR for decoder adaptation
WEIGHT_DECAY        = 1e-4       # Higher weight decay for better regularisation
LR_SCHEDULER        = "cosine"   # "cosine" or "step"
WARMUP_EPOCHS       = 5
GRAD_CLIP           = 1.0        # Max gradient norm
VAL_INTERVAL        = 2          # Validate more frequently
EARLY_STOP_PATIENCE = 50         # Increased patience to match extended epochs

# ─── Pretraining (Self-supervised) ────────────────────────────────────────────
PRETRAIN_EPOCHS     = 20         # Reduced pretrain epochs
PRETRAIN_LR         = 1e-4
PRETRAIN_BATCH_SIZE = 2

# ─── Inference ────────────────────────────────────────────────────────────────
INFERENCE_THRESHOLD = 0.5    # Sigmoid threshold for binary prediction

# Create directories
for d in [OUTPUT_DIR, CHECKPOINT_DIR, LOG_DIR]:
    os.makedirs(d, exist_ok=True)
