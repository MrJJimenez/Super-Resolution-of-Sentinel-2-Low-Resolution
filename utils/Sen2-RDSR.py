"""
Sen2-RDSR: Residual Dense Super-Resolution Network for Sentinel-2 data
PyTorch implementation of a residual dense convolutional neural network
for single-image super-resolution of Sentinel-2 low resolution bands.

This implementation is inspired by:
- DSen2Net architecture
- Residual Dense Networks (RDN) for super-resolution
- UNet implementation
"""

import torch
from torch import nn
from torch.nn import functional as F


class DenseLayer(nn.Module):
    """
    Dense layer within a residual dense block.
    Each layer takes all previous layers as input.
    """
    
    def __init__(self, in_chans: int, growth_rate: int, drop_prob: float = 0.0):
        """
        Args:
            in_chans: Number of input channels
            growth_rate: Number of new channels this layer adds
            drop_prob: Dropout probability
        """
        super().__init__()
        
        self.layer = nn.Sequential(
            nn.Conv2d(in_chans, growth_rate, kernel_size=3, padding=1, bias=True),
            nn.ReLU(inplace=True),
            nn.Dropout2d(drop_prob) if drop_prob > 0 else nn.Identity()
        )
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Input tensor
        Returns:
            Concatenation of input and new features
        """
        new_features = self.layer(x)
        return torch.cat([x, new_features], dim=1)


class ResidualDenseBlock(nn.Module):
    """
    Residual Dense Block (RDB) containing multiple dense layers
    with local residual learning.
    """
    
    def __init__(self, in_chans: int, growth_rate: int = 32, num_layers: int = 6, 
                 drop_prob: float = 0.0, res_scale: float = 0.2):
        """
        Args:
            in_chans: Number of input channels
            growth_rate: Growth rate for dense layers
            num_layers: Number of dense layers in the block
            drop_prob: Dropout probability
            res_scale: Residual scaling factor
        """
        super().__init__()
        
        self.res_scale = res_scale
        
        # Dense layers
        self.dense_layers = nn.ModuleList()
        for i in range(num_layers):
            layer_in_chans = in_chans + i * growth_rate
            self.dense_layers.append(DenseLayer(layer_in_chans, growth_rate, drop_prob))
        
        # Local feature fusion
        self.lff = nn.Conv2d(in_chans + num_layers * growth_rate, in_chans, 
                            kernel_size=1, padding=0, bias=True)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Input tensor of shape (N, in_chans, H, W)
        Returns:
            Output tensor of shape (N, in_chans, H, W)
        """
        identity = x
        
        # Pass through dense layers
        for layer in self.dense_layers:
            x = layer(x)
        
        # Local feature fusion
        x = self.lff(x)
        
        # Local residual learning
        return identity + x * self.res_scale


class Sen2RDSR(nn.Module):
    """
    Sen2-RDSR: Residual Dense Super-Resolution Network for Sentinel-2 data
    
    This model performs super-resolution of Sentinel-2 low resolution bands using
    a residual dense convolutional neural network architecture.
    """
    
    def __init__(
        self,
        input_shapes: tuple,
        num_rdb_blocks: int = 8,
        growth_rate: int = 32,
        rdb_layers: int = 6,
        feature_size: int = 64,
        drop_prob: float = 0.0,
        res_scale: float = 0.2,
        global_res_scale: float = 1.0
    ):
        """
        Args:
            input_shapes: Tuple of input shapes for different resolutions
                         e.g., ((4, H, W), (6, H//2, W//2)) for 10m and 20m bands
                         or ((4, H, W), (6, H//2, W//2), (2, H//6, W//6)) for 10m, 20m, 60m
            num_rdb_blocks: Number of residual dense blocks
            growth_rate: Growth rate for dense layers
            rdb_layers: Number of dense layers per RDB block
            feature_size: Number of feature channels
            drop_prob: Dropout probability
            res_scale: Local residual scaling factor
            global_res_scale: Global residual scaling factor
        """
        super().__init__()
        
        self.input_shapes = input_shapes
        self.num_inputs = len(input_shapes)
        self.global_res_scale = global_res_scale
        
        # Calculate total input channels after concatenation
        total_input_chans = sum(shape[0] for shape in input_shapes)
        
        # Initial feature extraction
        self.shallow_feature_extraction = nn.Sequential(
            nn.Conv2d(total_input_chans, feature_size, kernel_size=3, padding=1, bias=True),
            nn.ReLU(inplace=True)
        )
        
        # Residual Dense Blocks
        self.rdb_blocks = nn.ModuleList([
            ResidualDenseBlock(
                in_chans=feature_size,
                growth_rate=growth_rate,
                num_layers=rdb_layers,
                drop_prob=drop_prob,
                res_scale=res_scale
            ) for _ in range(num_rdb_blocks)
        ])
        
        # Global Feature Fusion
        self.gff = nn.Sequential(
            nn.Conv2d(feature_size * num_rdb_blocks, feature_size, kernel_size=1, padding=0, bias=True),
            nn.Conv2d(feature_size, feature_size, kernel_size=3, padding=1, bias=True)
        )
        
        # Final reconstruction
        # Output channels should match the target resolution
        output_chans = input_shapes[-1][0]  # Use the last input's channels as target
        self.reconstruction = nn.Conv2d(feature_size, output_chans, kernel_size=3, padding=1, bias=True)
        
    def _interpolate_inputs(self, inputs: list) -> torch.Tensor:
        """
        Interpolate all inputs to the highest resolution and concatenate them.
        
        Args:
            inputs: List of input tensors at different resolutions
        Returns:
            Concatenated tensor at highest resolution
        """
        if len(inputs) != self.num_inputs:
            raise ValueError(f"Expected {self.num_inputs} inputs, got {len(inputs)}")
        
        # Find the highest resolution (assuming first input is highest resolution)
        target_size = inputs[0].shape[2:]
        
        interpolated = []
        for i, inp in enumerate(inputs):
            if inp.shape[2:] != target_size:
                # Upsample to target resolution using bilinear interpolation
                inp_interp = F.interpolate(inp, size=target_size, mode='bilinear', align_corners=False)
                interpolated.append(inp_interp)
            else:
                interpolated.append(inp)
        
        return torch.cat(interpolated, dim=1)
    
    def forward(self, *inputs) -> torch.Tensor:
        """
        Forward pass of the Sen2-RDSR model.
        
        Args:
            *inputs: Variable number of input tensors at different resolutions
                    Expected order: highest resolution first (e.g., 10m, 20m, 60m)
        Returns:
            Super-resolved output tensor at the target resolution
        """
        # Concatenate inputs at highest resolution
        x = self._interpolate_inputs(list(inputs))
        
        # Shallow feature extraction
        shallow_features = self.shallow_feature_extraction(x)
        
        # Pass through residual dense blocks
        rdb_outputs = []
        rdb_input = shallow_features
        
        for rdb in self.rdb_blocks:
            rdb_output = rdb(rdb_input)
            rdb_outputs.append(rdb_output)
            rdb_input = rdb_output
        
        # Global feature fusion
        global_features = torch.cat(rdb_outputs, dim=1)
        global_features = self.gff(global_features)
        
        # Global residual learning
        enhanced_features = shallow_features + global_features * self.global_res_scale
        
        # Final reconstruction
        output = self.reconstruction(enhanced_features)
        
        # Add global residual connection to the target input (typically the lowest resolution)
        target_input = inputs[-1]  # Use last input as target for residual connection
        if target_input.shape[2:] != output.shape[2:]:
            target_input = F.interpolate(target_input, size=output.shape[2:], 
                                       mode='bilinear', align_corners=False)
        
        return output + target_input


def create_sen2_rdsr_model(
    model_type: str = "2x",  # "2x" for 20m->10m, "6x" for 60m->10m
    deep: bool = False,
    **kwargs
) -> Sen2RDSR:
    """
    Factory function to create Sen2-RDSR models with different configurations.
    
    Args:
        model_type: Type of super-resolution ("2x" or "6x")
        deep: Whether to use deep configuration
        **kwargs: Additional arguments for the model
    Returns:
        Configured Sen2-RDSR model
    """
    
    if model_type == "2x":
        # 20m -> 10m super-resolution
        input_shapes = ((4, None, None), (6, None, None))  # 10m and 20m bands
    elif model_type == "6x":
        # 60m -> 10m super-resolution  
        input_shapes = ((4, None, None), (6, None, None), (2, None, None))  # 10m, 20m, 60m bands
    else:
        raise ValueError(f"Unsupported model_type: {model_type}")
    
    if deep:
        # Deep configuration
        default_config = {
            "num_rdb_blocks": 16,
            "growth_rate": 32,
            "rdb_layers": 8,
            "feature_size": 128,
            "drop_prob": 0.1,
            "res_scale": 0.2,
            "global_res_scale": 1.0
        }
    else:
        # Standard configuration
        default_config = {
            "num_rdb_blocks": 8,
            "growth_rate": 32,
            "rdb_layers": 6,
            "feature_size": 64,
            "drop_prob": 0.0,
            "res_scale": 0.2,
            "global_res_scale": 1.0
        }
    
    # Update with any provided kwargs
    default_config.update(kwargs)
    
    return Sen2RDSR(input_shapes=input_shapes, **default_config)


# Example usage:
if __name__ == "__main__":
    # Create a 2x super-resolution model (20m -> 10m)
    model_2x = create_sen2_rdsr_model("2x", deep=False)
    
    # Create sample inputs
    input_10m = torch.randn(1, 4, 128, 128)   # 10m resolution bands
    input_20m = torch.randn(1, 6, 64, 64)     # 20m resolution bands
    
    # Forward pass
    output = model_2x(input_10m, input_20m)
    print(f"2x Model output shape: {output.shape}")
    
    # Create a 6x super-resolution model (60m -> 10m)
    model_6x = create_sen2_rdsr_model("6x", deep=True)
    
    # Create sample inputs
    input_60m = torch.randn(1, 2, 22, 22)     # 60m resolution bands
    
    # Forward pass
    output_6x = model_6x(input_10m, input_20m, input_60m)
    print(f"6x Model output shape: {output_6x.shape}")
    
    # Print model parameters
    total_params = sum(p.numel() for p in model_2x.parameters())
    trainable_params = sum(p.numel() for p in model_2x.parameters() if p.requires_grad)
    print(f"2x Model - Total parameters: {total_params:,}")
    print(f"2x Model - Trainable parameters: {trainable_params:,}")


"""
==================================================================================
                             Sen2-RDSR Model Documentation
==================================================================================

OVERVIEW:
--------
Sen2-RDSR (Sentinel-2 Residual Dense Super-Resolution) is a PyTorch implementation
of a residual dense convolutional neural network designed for super-resolution of
Sentinel-2 satellite imagery low resolution bands.

ARCHITECTURE:
------------
The model is based on Residual Dense Networks (RDN) and consists of:

1. **Shallow Feature Extraction**: Initial convolution to extract low-level features
2. **Residual Dense Blocks (RDB)**: Core building blocks with dense connections
3. **Global Feature Fusion**: Combines features from all RDB blocks
4. **Reconstruction Layer**: Final convolution to produce output
5. **Global Residual Connection**: Skip connection from input to output

KEY FEATURES:
------------
- **Dense Connections**: Each layer in RDB connects to all subsequent layers
- **Local Residual Learning**: Residual connections within each RDB block
- **Global Residual Learning**: Skip connection from input to final output
- **Multi-Resolution Input**: Handles inputs at different spatial resolutions
- **Flexible Architecture**: Configurable depth and complexity

USAGE EXAMPLES:
--------------

1. **Basic 2x Super-Resolution (20m -> 10m):**
```python
from utils.Sen2_RDSR import create_sen2_rdsr_model
import torch

# Create model
model = create_sen2_rdsr_model("2x", deep=False)

# Prepare inputs
input_10m = torch.randn(1, 4, 128, 128)  # 4 bands at 10m resolution
input_20m = torch.randn(1, 6, 64, 64)    # 6 bands at 20m resolution

# Forward pass
enhanced_20m = model(input_10m, input_20m)
print(f"Output shape: {enhanced_20m.shape}")  # [1, 6, 128, 128]
```

2. **Deep 6x Super-Resolution (60m -> 10m):**
```python
# Create deep model for 60m -> 10m super-resolution
model = create_sen2_rdsr_model("6x", deep=True)

# Prepare inputs
input_10m = torch.randn(1, 4, 128, 128)  # 4 bands at 10m resolution
input_20m = torch.randn(1, 6, 64, 64)    # 6 bands at 20m resolution  
input_60m = torch.randn(1, 2, 22, 22)    # 2 bands at 60m resolution

# Forward pass
enhanced_60m = model(input_10m, input_20m, input_60m)
print(f"Output shape: {enhanced_60m.shape}")  # [1, 2, 128, 128]
```

3. **Custom Configuration:**
```python
# Create model with custom parameters
model = Sen2RDSR(
    input_shapes=((4, None, None), (6, None, None)),
    num_rdb_blocks=12,
    growth_rate=48,
    rdb_layers=8,
    feature_size=96,
    drop_prob=0.1,
    res_scale=0.2,
    global_res_scale=1.0
)
```

TRAINING INTEGRATION:
-------------------
The model can be integrated with the existing training pipeline:

```python
import torch.nn as nn
from torch.optim import Adam

# Create model
model = create_sen2_rdsr_model("2x", deep=False)

# Loss function (compatible with existing training)
criterion = nn.L1Loss()  # Mean Absolute Error

# Optimizer
optimizer = Adam(model.parameters(), lr=1e-4)

# Training step
def train_step(model, input_10m, input_20m, target_20m):
    optimizer.zero_grad()
    
    # Forward pass
    output = model(input_10m, input_20m)
    
    # Compute loss
    loss = criterion(output, target_20m)
    
    # Backward pass
    loss.backward()
    optimizer.step()
    
    return loss.item()
```

SENTINEL-2 BAND CONFIGURATION:
-----------------------------
- **10m bands (4 channels)**: B02 (Blue), B03 (Green), B04 (Red), B08 (NIR)
- **20m bands (6 channels)**: B05, B06, B07, B8A, B11, B12
- **60m bands (2 channels)**: B01 (Coastal aerosol), B09 (Water vapour)

MODEL VARIANTS:
--------------
1. **Standard Model**: 8 RDB blocks, 64 features, suitable for quick training
2. **Deep Model**: 16 RDB blocks, 128 features, higher capacity for better results

PERFORMANCE CHARACTERISTICS:
---------------------------
- **2x Standard**: ~2.2M parameters, good balance of speed and quality
- **6x Deep**: ~8.5M parameters, highest quality but slower training
- **Memory**: Scales with input resolution and batch size
- **Training**: Compatible with existing data loaders and training scripts

COMPARISON WITH DSen2Net:
------------------------
| Feature | DSen2Net | Sen2-RDSR |
|---------|----------|-----------|
| Framework | Keras/TF | PyTorch |
| Architecture | Simple ResNet | Residual Dense |
| Connections | Skip only | Dense + Skip |
| Feature Reuse | Limited | Extensive |
| Parameters | ~1M | ~2-8M |
| Capacity | Moderate | High |

IMPLEMENTATION NOTES:
-------------------
- Uses bilinear interpolation for resolution matching
- ReLU activation throughout the network  
- Optional dropout for regularization
- Residual scaling for training stability
- Compatible with existing data preprocessing pipeline

For more details, refer to the paper:
"Single-Image Super-Resolution of Sentinel-2 Low Resolution Bands with 
Residual Dense Convolutional Neural Networks"
"""
