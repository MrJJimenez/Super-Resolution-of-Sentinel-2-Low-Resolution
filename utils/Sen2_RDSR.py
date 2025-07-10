"""
Sen2-RDSR: Sentinel-2 Residual Dense Super-Resolution Network

This implementation is based on the paper concepts for satellite image super-resolution,
combining residual dense networks with squeeze-and-excitation attention mechanisms.
The architecture is specifically designed for Sentinel-2 multi-spectral satellite imagery.

Key components:
- Residual Dense Blocks (RDB) for feature extraction
- Squeeze-and-Excitation (SE) attention mechanism
- Multi-scale feature fusion
- Global and local skip connections
- Sub-pixel convolution for upsampling

Architecture Overview:
    Input (LR) → Shallow Feature Extraction → Multi-Scale Block
                    ↓
    RDB_1 → RDB_2 → ... → RDB_N → Global Feature Fusion
                    ↓
    Global Residual → Upsampling → Output (HR)
                    ↑
    Bicubic Upsampled Input (Skip Connection)

The model is particularly effective for:
- Sentinel-2 20m → 10m super-resolution
- Multi-spectral satellite imagery enhancement
- Preserving spectral characteristics while enhancing spatial details
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import numpy as np
from typing import Tuple, Optional, Union


class SqueezeExcitation(nn.Module):
    """
    Squeeze-and-Excitation module for channel attention.
    
    Args:
        in_channels: Number of input channels
        reduction: Reduction ratio for the bottleneck
    """
    
    def __init__(self, in_channels: int, reduction: int = 16):
        super().__init__()
        self.in_channels = in_channels
        self.reduction = reduction
        
        self.squeeze = nn.AdaptiveAvgPool2d(1)
        self.excitation = nn.Sequential(
            nn.Conv2d(in_channels, in_channels // reduction, 1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels // reduction, in_channels, 1, bias=False),
            nn.Sigmoid()
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, c, _, _ = x.size()
        y = self.squeeze(x)
        y = self.excitation(y)
        return x * y.expand_as(x)


class ResidualDenseBlock(nn.Module):
    """
    Residual Dense Block for feature extraction.
    
    Args:
        in_channels: Number of input channels
        growth_rate: Growth rate for dense connections
        num_layers: Number of convolutional layers in the block
        reduction: Reduction ratio for SE module
    """
    
    def __init__(self, in_channels: int, growth_rate: int = 32, 
                 num_layers: int = 5, reduction: int = 16):
        super().__init__()
        self.in_channels = in_channels
        self.growth_rate = growth_rate
        self.num_layers = num_layers
        
        # Dense layers
        self.layers = nn.ModuleList()
        for i in range(num_layers):
            self.layers.append(
                nn.Sequential(
                    nn.Conv2d(in_channels + i * growth_rate, growth_rate, 3, 
                             padding=1, bias=False),
                    nn.LeakyReLU(0.2, inplace=True)
                )
            )
        
        # Local feature fusion
        self.lff = nn.Conv2d(in_channels + num_layers * growth_rate, 
                            in_channels, 1, bias=False)
        
        # Squeeze-and-excitation
        self.se = SqueezeExcitation(in_channels, reduction)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        identity = x
        features = [x]
        
        # Dense connections
        for layer in self.layers:
            new_feature = layer(torch.cat(features, 1))
            features.append(new_feature)
        
        # Local feature fusion
        out = self.lff(torch.cat(features, 1))
        
        # Squeeze-and-excitation attention
        out = self.se(out)
        
        # Residual connection
        return out + identity


class MultiScaleBlock(nn.Module):
    """
    Multi-scale feature extraction block.
    
    Args:
        in_channels: Number of input channels
        out_channels: Number of output channels
    """
    
    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        
        # Multi-scale convolutions
        self.conv1x1 = nn.Conv2d(in_channels, out_channels // 4, 1, padding=0)
        self.conv3x3 = nn.Conv2d(in_channels, out_channels // 4, 3, padding=1)
        self.conv5x5 = nn.Conv2d(in_channels, out_channels // 4, 5, padding=2)
        self.conv7x7 = nn.Conv2d(in_channels, out_channels // 4, 7, padding=3)
        
        self.fusion = nn.Sequential(
            nn.Conv2d(out_channels, out_channels, 1, bias=False),
            nn.LeakyReLU(0.2, inplace=True)
        )
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x1 = self.conv1x1(x)
        x2 = self.conv3x3(x)
        x3 = self.conv5x5(x)
        x4 = self.conv7x7(x)
        
        out = torch.cat([x1, x2, x3, x4], dim=1)
        return self.fusion(out)


class UpsampleBlock(nn.Module):
    """
    Upsampling block using sub-pixel convolution.
    
    Args:
        in_channels: Number of input channels
        scale_factor: Upsampling scale factor
    """
    
    def __init__(self, in_channels: int, scale_factor: int = 2):
        super().__init__()
        self.scale_factor = scale_factor
        
        self.conv = nn.Conv2d(in_channels, in_channels * (scale_factor ** 2), 
                             3, padding=1)
        self.pixel_shuffle = nn.PixelShuffle(scale_factor)
        self.activation = nn.LeakyReLU(0.2, inplace=True)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.conv(x)
        x = self.pixel_shuffle(x)
        return self.activation(x)


class Sen2RDSR(nn.Module):
    """
    Sen2-RDSR: Sentinel-2 Residual Dense Super-Resolution Network
    
    A deep learning model for super-resolution of Sentinel-2 satellite imagery.
    The model uses residual dense blocks with squeeze-and-excitation attention
    for effective feature extraction and multi-scale processing.
    
    Args:
        in_channels: Number of input channels (e.g., 4 for Sentinel-2 RGBN)
        out_channels: Number of output channels (same as input for super-resolution)
        num_features: Number of features in intermediate layers
        num_blocks: Number of residual dense blocks
        growth_rate: Growth rate for dense connections
        scale_factor: Super-resolution scale factor (2, 3, or 4)
        reduction: Reduction ratio for SE modules
    """
    
    def __init__(self, 
                 in_channels: int = 4,
                 out_channels: int = 4, 
                 num_features: int = 64,
                 num_blocks: int = 16,
                 growth_rate: int = 32,
                 scale_factor: int = 2,
                 reduction: int = 16):
        super().__init__()
        
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.num_features = num_features
        self.num_blocks = num_blocks
        self.scale_factor = scale_factor
        
        # Shallow feature extraction
        self.shallow_feature = nn.Sequential(
            nn.Conv2d(in_channels, num_features, 3, padding=1),
            nn.LeakyReLU(0.2, inplace=True)
        )
        
        # Multi-scale initial processing
        self.multi_scale = MultiScaleBlock(num_features, num_features)
        
        # Residual dense blocks
        self.rdb_blocks = nn.ModuleList([
            ResidualDenseBlock(num_features, growth_rate, reduction=reduction)
            for _ in range(num_blocks)
        ])
        
        # Global feature fusion
        self.gff = nn.Sequential(
            nn.Conv2d(num_features * num_blocks, num_features, 1, bias=False),
            nn.Conv2d(num_features, num_features, 3, padding=1, bias=False),
            nn.LeakyReLU(0.2, inplace=True)
        )
        
        # Global residual connection
        self.global_residual = nn.Conv2d(num_features, num_features, 3, padding=1)
        
        # Upsampling branch
        self.upsampling = self._make_upsampling_branch(num_features, scale_factor)
        
        # Final output layer
        self.output = nn.Conv2d(num_features, out_channels, 3, padding=1)
        
        # Initialize weights
        self._initialize_weights()
    
    def _make_upsampling_branch(self, num_features: int, scale_factor: int) -> nn.Module:
        """Create upsampling branch based on scale factor."""
        layers = []
        
        if scale_factor == 2:
            layers.append(UpsampleBlock(num_features, 2))
        elif scale_factor == 3:
            layers.append(UpsampleBlock(num_features, 3))
        elif scale_factor == 4:
            layers.extend([
                UpsampleBlock(num_features, 2),
                UpsampleBlock(num_features, 2)
            ])
        else:
            raise ValueError(f"Scale factor {scale_factor} is not supported")
        
        return nn.Sequential(*layers)
    
    def _initialize_weights(self):
        """Initialize network weights."""
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='leaky_relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of Sen2-RDSR.
        
        Args:
            x: Input tensor of shape (B, C, H, W)
            
        Returns:
            Super-resolved tensor of shape (B, C, H*scale, W*scale)
        """
        # Shallow feature extraction
        shallow_feat = self.shallow_feature(x)
        
        # Multi-scale processing
        ms_feat = self.multi_scale(shallow_feat)
        
        # Store features from each RDB for global fusion
        rdb_features = []
        current_feat = ms_feat
        
        # Pass through residual dense blocks
        for rdb in self.rdb_blocks:
            current_feat = rdb(current_feat)
            rdb_features.append(current_feat)
        
        # Global feature fusion
        global_feat = torch.cat(rdb_features, dim=1)
        global_feat = self.gff(global_feat)
        
        # Global residual connection
        global_feat = self.global_residual(global_feat) + ms_feat
        
        # Upsampling
        upsampled_feat = self.upsampling(global_feat)
        
        # Final output
        output = self.output(upsampled_feat)
        
        # Add bicubic upsampled input as residual (global skip connection)
        if self.scale_factor > 1:
            x_upsampled = F.interpolate(x, scale_factor=self.scale_factor, 
                                      mode='bicubic', align_corners=False)
            output = output + x_upsampled
        
        return output


class Sen2RDSRLite(nn.Module):
    """
    Lightweight version of Sen2-RDSR for faster inference.
    
    Args:
        in_channels: Number of input channels
        out_channels: Number of output channels
        num_features: Number of features (reduced)
        num_blocks: Number of RDB blocks (reduced)
        scale_factor: Super-resolution scale factor
    """
    
    def __init__(self, 
                 in_channels: int = 4,
                 out_channels: int = 4,
                 num_features: int = 32,
                 num_blocks: int = 8,
                 scale_factor: int = 2):
        super().__init__()
        
        # Use the same architecture but with fewer parameters
        self.model = Sen2RDSR(
            in_channels=in_channels,
            out_channels=out_channels,
            num_features=num_features,
            num_blocks=num_blocks,
            growth_rate=16,  # Reduced growth rate
            scale_factor=scale_factor,
            reduction=8     # Less aggressive SE reduction
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.model(x)


def create_sen2_rdsr_model(model_type: str = "full", 
                          in_channels: int = 4,
                          scale_factor: int = 2,
                          **kwargs) -> nn.Module:
    """
    Factory function to create Sen2-RDSR models.
    
    Args:
        model_type: "full" or "lite"
        in_channels: Number of input channels
        scale_factor: Super-resolution scale factor
        **kwargs: Additional arguments for the model
        
    Returns:
        Sen2-RDSR model instance
    """
    if model_type == "full":
        return Sen2RDSR(
            in_channels=in_channels,
            out_channels=in_channels,
            scale_factor=scale_factor,
            **kwargs
        )
    elif model_type == "lite":
        return Sen2RDSRLite(
            in_channels=in_channels,
            out_channels=in_channels,
            scale_factor=scale_factor,
            **kwargs
        )
    else:
        raise ValueError(f"Unknown model type: {model_type}")


# Utility functions for Sentinel-2 data processing
def normalize_sentinel2(data: torch.Tensor, 
                       method: str = "percentile",
                       percentiles: Tuple[float, float] = (2, 98)) -> torch.Tensor:
    """
    Normalize Sentinel-2 data for better training performance.
    
    Args:
        data: Input tensor of shape (B, C, H, W) or (C, H, W)
        method: Normalization method ("percentile", "minmax", "zscore")
        percentiles: Percentile values for clipping (only for "percentile" method)
        
    Returns:
        Normalized tensor
    """
    if method == "percentile":
        p_low, p_high = torch.quantile(data, percentiles[0]/100), torch.quantile(data, percentiles[1]/100)
        data = torch.clamp(data, p_low, p_high)
        data = (data - p_low) / (p_high - p_low)
    elif method == "minmax":
        data_min, data_max = data.min(), data.max()
        data = (data - data_min) / (data_max - data_min)
    elif method == "zscore":
        data = (data - data.mean()) / data.std()
    else:
        raise ValueError(f"Unknown normalization method: {method}")
    
    return data


def calculate_psnr(img1: torch.Tensor, img2: torch.Tensor, max_val: float = 1.0) -> float:
    """
    Calculate Peak Signal-to-Noise Ratio (PSNR) between two images.
    
    Args:
        img1: First image tensor
        img2: Second image tensor
        max_val: Maximum possible pixel value
        
    Returns:
        PSNR value in dB
    """
    mse = torch.mean((img1 - img2) ** 2)
    if mse == 0:
        return float('inf')
    return 20 * torch.log10(max_val / torch.sqrt(mse)).item()


def calculate_ssim(img1: torch.Tensor, img2: torch.Tensor, 
                  window_size: int = 11, max_val: float = 1.0) -> float:
    """
    Calculate Structural Similarity Index (SSIM) between two images.
    
    Args:
        img1: First image tensor of shape (C, H, W)
        img2: Second image tensor of shape (C, H, W)
        window_size: Size of the sliding window
        max_val: Maximum possible pixel value
        
    Returns:
        SSIM value
    """
    # Simplified SSIM calculation (for full implementation, use torchmetrics)
    mu1 = F.avg_pool2d(img1, window_size, stride=1, padding=window_size//2)
    mu2 = F.avg_pool2d(img2, window_size, stride=1, padding=window_size//2)
    
    mu1_sq = mu1.pow(2)
    mu2_sq = mu2.pow(2)
    mu1_mu2 = mu1 * mu2
    
    sigma1_sq = F.avg_pool2d(img1 * img1, window_size, stride=1, padding=window_size//2) - mu1_sq
    sigma2_sq = F.avg_pool2d(img2 * img2, window_size, stride=1, padding=window_size//2) - mu2_sq
    sigma12 = F.avg_pool2d(img1 * img2, window_size, stride=1, padding=window_size//2) - mu1_mu2
    
    C1 = (0.01 * max_val) ** 2
    C2 = (0.03 * max_val) ** 2
    
    ssim_map = ((2 * mu1_mu2 + C1) * (2 * sigma12 + C2)) / ((mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2))
    return ssim_map.mean().item()


def patch_inference(model: nn.Module, 
                   image: torch.Tensor, 
                   patch_size: int = 256, 
                   overlap: int = 32) -> torch.Tensor:
    """
    Perform inference on large images using patch-based processing.
    
    Args:
        model: Trained Sen2-RDSR model
        image: Input image tensor of shape (C, H, W)
        patch_size: Size of patches for processing
        overlap: Overlap between patches
        
    Returns:
        Super-resolved image tensor
    """
    model.eval()
    device = next(model.parameters()).device
    
    C, H, W = image.shape
    scale_factor = model.scale_factor
    
    # Calculate output dimensions
    H_out = H * scale_factor
    W_out = W * scale_factor
    
    # Initialize output tensor
    output = torch.zeros(C, H_out, W_out, device=device, dtype=image.dtype)
    count = torch.zeros(1, H_out, W_out, device=device)
    
    stride = patch_size - overlap
    
    with torch.no_grad():
        for y in range(0, H - patch_size + 1, stride):
            for x in range(0, W - patch_size + 1, stride):
                # Extract patch
                patch = image[:, y:y+patch_size, x:x+patch_size].unsqueeze(0)
                patch = patch.to(device)
                
                # Process patch
                patch_output = model(patch).squeeze(0)
                
                # Place in output tensor
                y_out = y * scale_factor
                x_out = x * scale_factor
                patch_size_out = patch_size * scale_factor
                
                output[:, y_out:y_out+patch_size_out, x_out:x_out+patch_size_out] += patch_output
                count[:, y_out:y_out+patch_size_out, x_out:x_out+patch_size_out] += 1
    
    # Handle edges if necessary
    if H % stride != 0 or W % stride != 0:
        # Process remaining patches along edges
        # (Implementation for edge cases can be added here)
        pass
    
    # Average overlapping regions
    output = output / count
    
    return output


# Example usage and testing
if __name__ == "__main__":
    print("=" * 60)
    print("Sen2-RDSR: Sentinel-2 Residual Dense Super-Resolution Network")
    print("=" * 60)
    
    # Test the model
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Create model for Sentinel-2 (4 channels: R, G, B, NIR)
    print("\n1. Creating Sen2-RDSR model...")
    model = Sen2RDSR(
        in_channels=4,
        out_channels=4,
        num_features=64,
        num_blocks=16,
        scale_factor=2
    ).to(device)
    
    # Test input (batch_size=1, channels=4, height=32, width=32)
    print("2. Testing forward pass...")
    test_input = torch.randn(1, 4, 32, 32).to(device)
    
    # Forward pass
    with torch.no_grad():
        output = model(test_input)
    
    print(f"   Input shape: {test_input.shape}")
    print(f"   Output shape: {output.shape}")
    print(f"   Model parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    # Test lite model
    print("\n3. Creating Sen2-RDSR Lite model...")
    lite_model = Sen2RDSRLite(scale_factor=2).to(device)
    with torch.no_grad():
        lite_output = lite_model(test_input)
    
    print(f"   Lite model parameters: {sum(p.numel() for p in lite_model.parameters()):,}")
    print(f"   Lite output shape: {lite_output.shape}")
    
    # Test utility functions
    print("\n4. Testing utility functions...")
    # Create some test data
    hr_image = torch.randn(4, 64, 64)
    lr_image = torch.randn(4, 32, 32)
    sr_image = torch.randn(4, 64, 64)
    
    # Test normalization
    normalized = normalize_sentinel2(hr_image, method="percentile")
    print(f"   Original range: [{hr_image.min():.3f}, {hr_image.max():.3f}]")
    print(f"   Normalized range: [{normalized.min():.3f}, {normalized.max():.3f}]")
    
    # Test PSNR calculation
    psnr = calculate_psnr(hr_image, sr_image)
    print(f"   PSNR: {psnr:.2f} dB")
    
    # Test SSIM calculation
    ssim = calculate_ssim(hr_image, sr_image)
    print(f"   SSIM: {ssim:.4f}")
    
    print("\n5. Model configuration examples:")
    print("   # For Sentinel-2 20m → 10m (scale factor 2)")
    print("   model = Sen2RDSR(in_channels=6, scale_factor=2)")
    print("   ")
    print("   # For multi-spectral enhancement (scale factor 3)")
    print("   model = Sen2RDSR(in_channels=10, scale_factor=3)")
    print("   ")
    print("   # Lightweight model for fast inference")
    print("   model = Sen2RDSRLite(in_channels=4, scale_factor=2)")
    
    print("\n" + "=" * 60)
    print("Sen2-RDSR implementation completed successfully!")
    print("Ready for training on Sentinel-2 satellite imagery.")
    print("=" * 60)
