# =============================================================================
# dataset.py
# PANTHER Task 1 – Dataset loading, preprocessing, and MONAI data pipeline
# =============================================================================

import os
import glob
import random
from sklearn.model_selection import train_test_split

from monai.data import Dataset, CacheDataset, DataLoader
from monai.transforms import (
    Compose,
    LoadImaged,
    EnsureChannelFirstd,
    Spacingd,
    Orientationd,
    ScaleIntensityRangePercentilesd,
    NormalizeIntensityd,
    CropForegroundd,
    RandCropByPosNegLabeld,
    RandFlipd,
    RandRotate90d,
    RandScaleIntensityd,
    RandShiftIntensityd,
    RandGaussianNoised,
    RandAffined,
    ToTensord,
    EnsureTyped,
    AsDiscreted,
    SpatialPadd,
)

import config

from monai.transforms import (
    RandSpatialCropd,    # ✅ add this to the existing import block

)


# ─── Build file list dictionaries ─────────────────────────────────────────────

def get_labeled_data_dicts(task_dir=config.LABELED_DIR):
    """
    Returns a list of dicts: [{"image": path, "label": path}, ...]
    """
    image_paths = sorted(glob.glob(os.path.join(task_dir, "ImagesTr", "*.mha")))
    mask_paths = sorted(glob.glob(os.path.join(task_dir, "LabelsTr", "*.mha")))

    assert len(image_paths) == len(mask_paths), (
        f"Mismatch: {len(image_paths)} images vs {len(mask_paths)} masks in {task_dir}"
    )
    assert len(image_paths) > 0, f"No labeled images found in {task_dir}"

    data_dicts = [{"image": img, "label": lbl}
                  for img, lbl in zip(image_paths, mask_paths)]

    print(f"[Dataset] Found {len(data_dicts)} labeled samples")
    return data_dicts


def get_unlabeled_data_dicts():
    # Search both directly in folder AND in subfolders
    image_paths = sorted(glob.glob(os.path.join(config.UNLABELED_DIR, "*.mha")))

    # If still empty, try recursive search
    if len(image_paths) == 0:
        image_paths = sorted(glob.glob(os.path.join(config.UNLABELED_DIR, "**", "*.mha"), recursive=True))

    # Debug print so you can see what path is being searched
    print(f"[Dataset] Searching in: {config.UNLABELED_DIR}")
    print(f"[Dataset] Found {len(image_paths)} unlabeled samples")

    data_dicts = [{"image": p} for p in image_paths]
    return data_dicts

def split_data(data_dicts, train_ratio=config.TRAIN_RATIO, seed=config.RANDOM_SEED):
    """Split labeled data into train / validation sets."""
    train_data, val_data = train_test_split(
        data_dicts,
        test_size=(1.0 - train_ratio),
        random_state=seed,
        shuffle=True
    )
    print(f"[Dataset] Train: {len(train_data)}  |  Val: {len(val_data)}")
    return train_data, val_data


# ─── Transforms ───────────────────────────────────────────────────────────────



def get_train_transforms():
    """Full augmentation pipeline for training."""
    return Compose([
        # ── Load & basic setup ──────────────────────────────────────────────
        LoadImaged(keys=["image", "label"], reader="ITKReader"),
        EnsureChannelFirstd(keys=["image", "label"]),
        EnsureTyped(keys=["image", "label"]),

        # ── Resample to uniform voxel spacing ───────────────────────────────
        Spacingd(
            keys=["image", "label"],
            pixdim=config.TARGET_SPACING,
            mode=("bilinear", "nearest")   # bilinear for images, nearest for masks
        ),

        # ── Canonical orientation (RAS) ──────────────────────────────────────
        Orientationd(keys=["image", "label"], axcodes="RAS", labels=None),

        # ── Intensity normalization (z-score per volume) ─────────────────────
        NormalizeIntensityd(keys=["image"], nonzero=True, channel_wise=True),

        # ── Crop tightly around foreground (removes empty border) ────────────
        CropForegroundd(keys=["image", "label"], source_key="image"),

        # ── Pad to at least patch size if volume is smaller ──────────────────
        SpatialPadd(keys=["image", "label"], spatial_size=config.PATCH_SIZE),

        # ── Random patch extraction centered on tumor (pos:neg = 1:1) ────────
        RandCropByPosNegLabeld(
            keys=["image", "label"],
            label_key="label",
            spatial_size=config.PATCH_SIZE,
            pos=1,
            neg=1,
            num_samples=2,    # Extract 2 patches per volume per step
            image_key="image",
            image_threshold=0,
        ),

        # ── Spatial augmentations ─────────────────────────────────────────────
        RandFlipd(keys=["image", "label"], prob=0.5, spatial_axis=0),
        RandFlipd(keys=["image", "label"], prob=0.5, spatial_axis=1),
        RandFlipd(keys=["image", "label"], prob=0.5, spatial_axis=2),
        RandRotate90d(keys=["image", "label"], prob=0.3, max_k=3),
        RandAffined(
            keys=["image", "label"],
            prob=0.3,
            rotate_range=(0.26, 0.26, 0.26),   # ±15 degrees
            scale_range=(0.15, 0.15, 0.15),    # ±15% scaling
            mode=("bilinear", "nearest"),
            padding_mode="border"
        ),

        # ── Intensity augmentations ───────────────────────────────────────────
        RandScaleIntensityd(keys=["image"], factors=0.1, prob=0.5),
        RandShiftIntensityd(keys=["image"], offsets=0.1, prob=0.5),
        RandGaussianNoised(keys=["image"], prob=0.3, mean=0.0, std=0.1),

        ToTensord(keys=["image", "label"]),
    ])


def get_val_transforms():
    """Minimal transforms for validation (no augmentation)."""
    return Compose([
        LoadImaged(keys=["image", "label"], reader="ITKReader"),
        EnsureChannelFirstd(keys=["image", "label"]),
        EnsureTyped(keys=["image", "label"]),
        Spacingd(
            keys=["image", "label"],
            pixdim=config.TARGET_SPACING,
            mode=("bilinear", "nearest")
        ),
        Orientationd(keys=["image", "label"], axcodes="RAS", labels=None),
        NormalizeIntensityd(keys=["image"], nonzero=True, channel_wise=True),
        CropForegroundd(keys=["image", "label"], source_key="image"),
        SpatialPadd(keys=["image", "label"], spatial_size=config.PATCH_SIZE),
        ToTensord(keys=["image", "label"]),
    ])


def get_pretrain_transforms():
    #Random spatial crop (no label needed)
    return Compose([
        LoadImaged(keys=["image"], reader="ITKReader"),
        EnsureChannelFirstd(keys=["image"]),
        EnsureTyped(keys=["image"]),
        Spacingd(keys=["image"], pixdim=config.TARGET_SPACING, mode="bilinear"),
        Orientationd(keys=["image"], axcodes="RAS", labels=None),
        NormalizeIntensityd(keys=["image"], nonzero=True, channel_wise=True),
        CropForegroundd(keys=["image"], source_key="image"),
        SpatialPadd(keys=["image"], spatial_size=config.PATCH_SIZE),   # pad if too small
        RandSpatialCropd(                                               # ✅ ADD THIS
            keys=["image"],
            roi_size=config.PATCH_SIZE,
            random_size=False
        ),
        ToTensord(keys=["image"]),
    ])


# ─── DataLoaders ──────────────────────────────────────────────────────────────

def get_train_val_loaders(task_dir=config.LABELED_DIR):
    """Build and return train and validation DataLoaders."""
    all_data              = get_labeled_data_dicts(task_dir=task_dir)
    train_data, val_data  = split_data(all_data)

    train_ds = CacheDataset(
        data=train_data,
        transform=get_train_transforms(),
        cache_rate=config.CACHE_RATE,
        num_workers=config.NUM_WORKERS
    )
    val_ds = CacheDataset(
        data=val_data,
        transform=get_val_transforms(),
        cache_rate=config.CACHE_RATE,
        num_workers=config.NUM_WORKERS
    )

    train_loader = DataLoader(
        train_ds,
        batch_size=config.BATCH_SIZE,
        shuffle=True,
        num_workers=config.NUM_WORKERS,
        pin_memory=True,
        persistent_workers=(config.NUM_WORKERS > 0)
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=1,
        shuffle=False,
        num_workers=0,          # Set to 0 for Windows to avoid 'shared file mapping' error
        pin_memory=False,       # Avoid MetaTensor double-pin crash
        persistent_workers=False
    )

    print(f"[DataLoader] Train batches: {len(train_loader)}  |  Val batches: {len(val_loader)}")
    return train_loader, val_loader


def get_pretrain_loader():
    """DataLoader for unlabeled data (self-supervised pretraining)."""
    unlabeled_data = get_unlabeled_data_dicts()

    pretrain_ds = CacheDataset(
        data=unlabeled_data,
        transform=get_pretrain_transforms(),
        cache_rate=0.5,             # Only cache half to save RAM
        num_workers=config.NUM_WORKERS
    )

    return DataLoader(
        pretrain_ds,
        batch_size=config.PRETRAIN_BATCH_SIZE,
        shuffle=True,
        num_workers=config.NUM_WORKERS,
        pin_memory=True,
        persistent_workers=True
    )
