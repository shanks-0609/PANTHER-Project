# =============================================================================
# model.py
# PANTHER Task 1 – 3D Attention U-Net architecture
#
# Architecture:
#   Encoder (4 downsampling levels)
#     → Bottleneck
#       → Decoder (4 upsampling levels) with Attention Gates on skip connections
#         → Output head (1×1×1 conv → softmax)
#
# Reference: Oktay et al., "Attention U-Net: Learning Where to Look for the
#            Pancreas," MIDL 2018.
# =============================================================================

import torch
import torch.nn as nn
import torch.nn.functional as F
import config


# ─── Building Blocks ──────────────────────────────────────────────────────────

class ConvBlock(nn.Module):
    """Two consecutive Conv3d → BatchNorm → ReLU layers (the U-Net 'double conv')."""

    def __init__(self, in_ch, out_ch, dropout=0.0):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv3d(in_ch, out_ch, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm3d(out_ch),
            nn.ReLU(inplace=True),
            nn.Dropout3d(p=dropout),
            nn.Conv3d(out_ch, out_ch, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm3d(out_ch),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.block(x)


class DownBlock(nn.Module):
    """MaxPool3d downsampling followed by a ConvBlock."""

    def __init__(self, in_ch, out_ch, dropout=0.0):
        super().__init__()
        self.pool    = nn.MaxPool3d(kernel_size=2, stride=2)
        self.conv    = ConvBlock(in_ch, out_ch, dropout=dropout)

    def forward(self, x):
        return self.conv(self.pool(x))


class UpBlock(nn.Module):
    """Transposed conv upsampling + concatenation with skip + ConvBlock."""

    def __init__(self, in_ch, skip_ch, out_ch, dropout=0.0):
        super().__init__()
        self.up   = nn.ConvTranspose3d(in_ch, in_ch // 2, kernel_size=2, stride=2)
        self.conv = ConvBlock(in_ch // 2 + skip_ch, out_ch, dropout=dropout)

    def forward(self, x, skip):
        x = self.up(x)

        # Handle size mismatch from rounding during downsampling
        if x.shape != skip.shape:
            x = F.interpolate(x, size=skip.shape[2:], mode="trilinear", align_corners=False)

        x = torch.cat([skip, x], dim=1)
        return self.conv(x)


class AttentionGate(nn.Module):
    """
    Soft attention gate for skip connections.
    Learns to suppress irrelevant background features before concatenation.

    g  = gating signal (from decoder, lower resolution)
    x  = skip connection (from encoder, same resolution as output)
    """

    def __init__(self, g_ch, x_ch, inter_ch):
        super().__init__()
        # 1×1×1 convolutions to compute compatibility score
        self.W_g = nn.Sequential(
            nn.Conv3d(g_ch, inter_ch, kernel_size=1, bias=True),
            nn.BatchNorm3d(inter_ch)
        )
        self.W_x = nn.Sequential(
            nn.Conv3d(x_ch, inter_ch, kernel_size=1, stride=2, bias=True),
            nn.BatchNorm3d(inter_ch)
        )
        self.psi = nn.Sequential(
            nn.Conv3d(inter_ch, 1, kernel_size=1, bias=True),
            nn.BatchNorm3d(1),
            nn.Sigmoid()
        )
        self.relu = nn.ReLU(inplace=True)

    def forward(self, g, x):
        # g: gating signal (coarser), x: skip connection (finer)
        g1  = self.W_g(g)
        x1  = self.W_x(x)

        # Upsample gating signal to match skip connection spatial size
        g1  = F.interpolate(g1, size=x1.shape[2:], mode="trilinear", align_corners=False)

        psi = self.relu(g1 + x1)
        psi = self.psi(psi)

        # Upsample attention map to match x
        psi = F.interpolate(psi, size=x.shape[2:], mode="trilinear", align_corners=False)

        return x * psi   # Element-wise reweighting of skip features


# ─── Full 3D Attention U-Net ──────────────────────────────────────────────────

class AttentionUNet3D(nn.Module):
    """
    3D Attention U-Net for volumetric pancreatic tumor segmentation.

    Args:
        in_channels  (int): Number of input channels (1 for grayscale MRI)
        out_channels (int): Number of output classes (2: background + tumor)
        features     (tuple): Feature channels at each encoder level
        dropout      (float): Dropout rate inside ConvBlocks
    """

    def __init__(
        self,
        in_channels  = config.IN_CHANNELS,
        out_channels = config.OUT_CHANNELS,
        features     = config.FEATURE_CHANNELS,
        dropout      = config.DROPOUT_RATE
    ):
        super().__init__()

        f = features   # shorthand: (32, 64, 128, 256, 512)

        # ── Encoder ──────────────────────────────────────────────────────────
        self.enc1 = ConvBlock(in_channels, f[0], dropout=dropout)   # 128→64
        self.enc2 = DownBlock(f[0], f[1], dropout=dropout)          # 64→32
        self.enc3 = DownBlock(f[1], f[2], dropout=dropout)          # 32→16
        self.enc4 = DownBlock(f[2], f[3], dropout=dropout)          # 16→8

        # ── Bottleneck ───────────────────────────────────────────────────────
        self.bottleneck = DownBlock(f[3], f[4], dropout=dropout)    # 8→4

        # ── Attention Gates ───────────────────────────────────────────────────
        # Gate signal comes from decoder (g), skip from encoder (x)
        self.att4 = AttentionGate(g_ch=f[4], x_ch=f[3], inter_ch=f[3] // 2)
        self.att3 = AttentionGate(g_ch=f[3], x_ch=f[2], inter_ch=f[2] // 2)
        self.att2 = AttentionGate(g_ch=f[2], x_ch=f[1], inter_ch=f[1] // 2)
        self.att1 = AttentionGate(g_ch=f[1], x_ch=f[0], inter_ch=f[0] // 2)

        # ── Decoder ──────────────────────────────────────────────────────────
        self.dec4 = UpBlock(f[4], f[3], f[3], dropout=dropout)
        self.dec3 = UpBlock(f[3], f[2], f[2], dropout=dropout)
        self.dec2 = UpBlock(f[2], f[1], f[1], dropout=dropout)
        self.dec1 = UpBlock(f[1], f[0], f[0], dropout=dropout)

        # ── Output head ───────────────────────────────────────────────────────
        self.output_conv = nn.Conv3d(f[0], out_channels, kernel_size=1)

        # Weight initialization
        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv3d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.BatchNorm3d):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)

    def forward(self, x):
        # ── Encoder forward pass ──────────────────────────────────────────────
        s1 = self.enc1(x)         # Skip connection 1
        s2 = self.enc2(s1)        # Skip connection 2
        s3 = self.enc3(s2)        # Skip connection 3
        s4 = self.enc4(s3)        # Skip connection 4

        # ── Bottleneck ───────────────────────────────────────────────────────
        b  = self.bottleneck(s4)

        # ── Decoder forward pass with attention-gated skip connections ────────
        s4_att = self.att4(g=b,  x=s4)
        d4     = self.dec4(b,  s4_att)

        s3_att = self.att3(g=d4, x=s3)
        d3     = self.dec3(d4, s3_att)

        s2_att = self.att2(g=d3, x=s2)
        d2     = self.dec2(d3, s2_att)

        s1_att = self.att1(g=d2, x=s1)
        d1     = self.dec1(d2, s1_att)

        # ── Output ────────────────────────────────────────────────────────────
        logits = self.output_conv(d1)  # Shape: (B, out_channels, D, H, W)
        return logits


# ─── Quick sanity check ───────────────────────────────────────────────────────

if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model  = AttentionUNet3D().to(device)

    dummy  = torch.randn(1, 1, *config.PATCH_SIZE).to(device)   # (B=1, C=1, D=128, H=128, W=64)
    output = model(dummy)

    print(f"Input  shape : {dummy.shape}")
    print(f"Output shape : {output.shape}")   # Expected: (1, 2, 128, 128, 64)

    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Trainable parameters: {total_params:,}")
