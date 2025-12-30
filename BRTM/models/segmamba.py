"""
SegMamba: Pure Mamba-Based U-Net Architecture for 3D Medical Image Segmentation
Combines Conv3D for local features with Mamba state-space blocks for global context.

Architecture Philosophy:
- Encoder: Hierarchical with Conv3D blocks (early stages) + Mamba (deep stages)
- Bottleneck: State-space model (Mamba) for long-range dependencies
- Decoder: Standard Conv3D with skip connections

References:
- U-Mamba: https://arxiv.org/abs/2401.04722
- Mamba: https://arxiv.org/abs/2312.00752
- nnU-Net: https://arxiv.org/abs/2011.00848
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Optional, Tuple


# Import Mamba SSM (state-space model) - REQUIRED
try:
    from mamba_ssm import Mamba
    MAMBA_AVAILABLE = True
    print("[SegMamba] Mamba SSM loaded successfully - Pure Mamba architecture enabled")
except ImportError:
    print("\n" + "="*80)
    print("⚠️  WARNING: mamba-ssm not available - using MOCK implementation for testing")
    print("   This is NOT the real state-space model - for testing only!")
    print("="*80 + "\n")
    try:
        import sys
        from pathlib import Path
        sys.path.insert(0, str(Path(__file__).parent))
        from mock_mamba import Mamba
        MAMBA_AVAILABLE = False
    except ImportError as e:
        print(f"Could not import mock mamba: {e}")
        raise ImportError(
            "Mamba SSM is required for SegMamba. Install with:\n"
            "  pip install mamba-ssm\n"
            "  pip install causal-conv1d\n"
            "Or follow: https://github.com/state-spaces/mamba"
        )




class Conv3DBlock(nn.Module):
    """
    Standard 3D Convolutional Block with BatchNorm and ReLU.
    
    This is the workhorse for local feature extraction in early encoder stages.
    
    Args:
        in_channels: Number of input channels
        out_channels: Number of output channels
        kernel_size: Convolution kernel size
        stride: Convolution stride
        padding: Convolution padding
    """
    
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int = 3,
        stride: int = 1,
        padding: int = 1
    ):
        super().__init__()
        
        self.conv = nn.Sequential(
            nn.Conv3d(in_channels, out_channels, kernel_size, stride, padding, bias=False),
            nn.BatchNorm3d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv3d(out_channels, out_channels, kernel_size, 1, padding, bias=False),
            nn.BatchNorm3d(out_channels),
            nn.ReLU(inplace=True)
        )
        
        # Residual connection
        self.residual = nn.Sequential(
            nn.Conv3d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False),
            nn.BatchNorm3d(out_channels)
        ) if in_channels != out_channels or stride != 1 else nn.Identity()
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.
        
        Args:
            x: Input tensor (B, C, D, H, W)
            
        Returns:
            Output tensor (B, C_out, D', H', W')
        """
        identity = self.residual(x)
        out = self.conv(x)
        return out + identity





class MambaBlock3D(nn.Module):
    """
    3D Mamba Block for State-Space Modeling.
    
    Mamba provides linear-time complexity for sequence modeling,
    making it superior to Transformers for 3D medical images.
    
    This block flattens 3D spatial dimensions into sequences for Mamba processing.
    
    Args:
        dim: Feature dimension
        d_state: State space dimension
        d_conv: Convolution dimension for local mixing
        expand: Expansion factor for internal dimensions
    """
    
    def __init__(
        self,
        dim: int,
        d_state: int = 16,
        d_conv: int = 4,
        expand: int = 2
    ):
        super().__init__()
        
        self.dim = dim
        
        # Mamba layer (uses either real or mock implementation)
        self.mamba = Mamba(
            d_model=dim,
            d_state=d_state,
            d_conv=d_conv,
            expand=expand
        )
        
        self.norm = nn.LayerNorm(dim)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.
        
        Args:
            x: Input tensor (B, C, D, H, W)
            
        Returns:
            Output tensor (B, C, D, H, W)
        """
        B, C, D, H, W = x.shape
        
        # Reshape to sequence: (B, D*H*W, C)
        x_seq = x.flatten(2).transpose(1, 2)
        
        # Normalize
        x_norm = self.norm(x_seq)
        
        # Apply Mamba (processes sequences with linear complexity)
        mamba_out = self.mamba(x_norm)
        
        # Residual connection
        x_seq = x_seq + mamba_out
        
        # Reshape back: (B, C, D, H, W)
        x_out = x_seq.transpose(1, 2).reshape(B, C, D, H, W)
        
        return x_out


class EncoderStage(nn.Module):
    """
    Encoder stage with Conv3D or Mamba blocks.
    
    Early stages use Conv3D for local features.
    Deep stages use Mamba for global context.
    
    Args:
        in_channels: Number of input channels
        out_channels: Number of output channels
        num_blocks: Number of blocks in this stage
        use_mamba: If True, uses Mamba blocks; else uses Conv3D
    """
    
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        num_blocks: int = 2,
        use_mamba: bool = False
    ):
        super().__init__()
        
        blocks = []
        
        for i in range(num_blocks):
            in_ch = in_channels if i == 0 else out_channels
            
            if use_mamba:
                # Use Mamba for global context
                if in_ch != out_channels:
                    blocks.append(nn.Conv3d(in_ch, out_channels, 1))
                blocks.append(MambaBlock3D(dim=out_channels))
            else:
                # Use standard Conv3D
                blocks.append(Conv3DBlock(in_ch, out_channels))
        
        self.blocks = nn.Sequential(*blocks)
        
        # Downsampling with channel projection
        self.downsample = nn.Conv3d(out_channels, out_channels * 2, kernel_size=2, stride=2)
    
    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass.
        
        Args:
            x: Input tensor
            
        Returns:
            Tuple of (downsampled output, skip connection)
        """
        skip = self.blocks(x)
        out = self.downsample(skip)
        return out, skip


class DecoderStage(nn.Module):
    """
    Decoder stage with upsampling and skip connections.
    
    Args:
        in_channels: Number of input channels
        skip_channels: Number of channels from skip connection
        out_channels: Number of output channels
        num_blocks: Number of convolutional blocks
    """
    
    def __init__(
        self,
        in_channels: int,
        skip_channels: int,
        out_channels: int,
        num_blocks: int = 2
    ):
        super().__init__()
        
        # Upsampling
        self.upsample = nn.ConvTranspose3d(
            in_channels, out_channels, kernel_size=2, stride=2
        )
        
        # Convolutional blocks after concatenation
        blocks = []
        for i in range(num_blocks):
            in_ch = out_channels + skip_channels if i == 0 else out_channels
            blocks.append(Conv3DBlock(in_ch, out_channels))
        
        self.blocks = nn.Sequential(*blocks)
    
    def forward(self, x: torch.Tensor, skip: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.
        
        Args:
            x: Input tensor from previous decoder stage
            skip: Skip connection from encoder
            
        Returns:
            Output tensor
        """
        # Upsample
        x = self.upsample(x)
        
        # Handle size mismatch (in case of odd dimensions)
        if x.shape != skip.shape:
            x = F.interpolate(x, size=skip.shape[2:], mode='trilinear', align_corners=False)
        
        # Concatenate skip connection
        x = torch.cat([x, skip], dim=1)
        
        # Apply convolutional blocks
        x = self.blocks(x)
        
        return x


class SegMamba(nn.Module):
    """
    SegMamba: Pure Mamba-Based 3D U-Net for Medical Image Segmentation.
    
    Architecture:
        - Input: (B, 4, D, H, W) - 4 MRI modalities
        - Encoder: 4 stages with increasing channels and Mamba in deep stages
        - Decoder: 4 stages with skip connections
        - Output: (B, num_classes, D, H, W) - segmentation logits
    
    Key Features:
        1. Early encoder stages use Conv3D for local features
        2. Deep encoder stages use Mamba for global context
        3. Decoder uses standard Conv3D with skip connections
        4. Optimized for single GPU training with optional gradient checkpointing
    
    Args:
        in_channels: Number of input modalities (default: 4 for BraTS)
        num_classes: Number of segmentation classes (default: 4)
        base_channels: Base number of channels (default: 32)
        encoder_depths: List of blocks per encoder stage
        use_checkpoint: Enable gradient checkpointing to save memory
    """
    
    def __init__(
        self,
        in_channels: int = 4,
        num_classes: int = 4,
        base_channels: int = 32,
        encoder_depths: List[int] = [2, 2, 2, 2],
        use_checkpoint: bool = False
    ):
        super().__init__()
        
        # Note: If MAMBA_AVAILABLE is False, mock implementation will be used
        if not MAMBA_AVAILABLE:
            print("⚠️  Using mock Mamba - for testing only!")

        
        self.in_channels = in_channels
        self.num_classes = num_classes
        self.use_checkpoint = use_checkpoint
        
        # Channel progression: 32 -> 64 -> 128 -> 256
        channels = [base_channels * (2 ** i) for i in range(len(encoder_depths))]
        
        # Initial convolution
        self.initial_conv = Conv3DBlock(in_channels, channels[0])
        
        # Encoder stages
        self.encoders = nn.ModuleList()
        for i in range(len(encoder_depths)):
            in_ch = channels[0] if i == 0 else channels[i-1] * 2  # Input from previous stage's downsampling
            out_ch = channels[i]  # Output channels for this stage
            num_blocks = encoder_depths[i]
            
            # Use Mamba in deeper stages (last 2 stages)
            use_mamba = (i >= len(encoder_depths) - 2)
            
            self.encoders.append(
                EncoderStage(
                    in_channels=in_ch,
                    out_channels=out_ch,
                    num_blocks=num_blocks,
                    use_mamba=use_mamba
                )
            )
        
        # Bottleneck (deepest layer with global context using Mamba)
        # Input from last encoder's downsampling
        bottleneck_in_ch = channels[-1] * 2
        bottleneck_ch = bottleneck_in_ch
        self.bottleneck = nn.Sequential(
            Conv3DBlock(bottleneck_in_ch, bottleneck_ch),
            MambaBlock3D(dim=bottleneck_ch),
            Conv3DBlock(bottleneck_ch, bottleneck_ch)
        )
        
        # Decoder stages
        self.decoders = nn.ModuleList()
        for i in range(len(encoder_depths) - 1, -1, -1):
            in_ch = bottleneck_ch if i == len(encoder_depths) - 1 else channels[i + 1]
            skip_ch = channels[i]
            out_ch = channels[i]
            
            self.decoders.append(
                DecoderStage(
                    in_channels=in_ch,
                    skip_channels=skip_ch,
                    out_channels=out_ch,
                    num_blocks=2
                )
            )
        
        # Final segmentation head
        self.segmentation_head = nn.Conv3d(channels[0], num_classes, kernel_size=1)
        
        # Initialize weights
        self._initialize_weights()
        
        print(f"[SegMamba] Initialized Pure Mamba architecture with {self.count_parameters():,} parameters")
        print(f"[SegMamba] Using Mamba state-space blocks for global context modeling")
    
    def _initialize_weights(self):
        """
        Initialize model weights using Kaiming initialization.
        """
        for m in self.modules():
            if isinstance(m, (nn.Conv3d, nn.ConvTranspose3d)):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm3d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.trunc_normal_(m.weight, std=0.02)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.
        
        Args:
            x: Input tensor (B, in_channels, D, H, W)
            
        Returns:
            Segmentation logits (B, num_classes, D, H, W)
        """
        # Initial convolution
        x = self.initial_conv(x)
        
        # Encoder with skip connections
        skips = []
        for encoder in self.encoders:
            x, skip = encoder(x)
            skips.append(skip)
        
        # Bottleneck
        x = self.bottleneck(x)
        
        # Decoder with skip connections
        for decoder, skip in zip(self.decoders, reversed(skips)):
            x = decoder(x, skip)
        
        # Segmentation head
        logits = self.segmentation_head(x)
        
        return logits
    
    def count_parameters(self) -> int:
        """
        Count total trainable parameters.
        
        Returns:
            Number of parameters
        """
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


if __name__ == "__main__":
    # Test model
    print("Testing SegMamba Pure Mamba architecture...")
    
    if not MAMBA_AVAILABLE:
        print("ERROR: Mamba SSM not installed. Install with: pip install mamba-ssm")
        exit(1)
    
    # Create model
    model = SegMamba(
        in_channels=4,
        num_classes=4,
        base_channels=32
    )
    
    # Test forward pass
    batch_size = 2
    x = torch.randn(batch_size, 4, 64, 64, 64)
    
    print(f"Input shape: {x.shape}")
    
    with torch.no_grad():
        output = model(x)
    
    print(f"Output shape: {output.shape}")
    print(f"Model parameters: {model.count_parameters():,}")
    print("✓ Pure Mamba architecture test successful!")
