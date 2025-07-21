import torch
import torch.nn as nn
import torch.nn.functional as F

# Dense Block (5 conv layers with dense connections)
class DenseBlock(nn.Module):
    def __init__(self, channels=32):
        super().__init__()
        self.layers = nn.ModuleList([
            nn.Conv2d(channels * i + channels, channels, 3, padding=1)
            for i in range(5)
        ])
        self.activation = nn.LeakyReLU(0.2, inplace=True)

    def forward(self, x):
        features = [x]
        for layer in self.layers:
            out = layer(torch.cat(features, 1))
            out = self.activation(out)
            features.append(out)
        return torch.cat(features, 1)[:, :x.shape[1], :, :]  # crop to input channels


# RRDB: Residual-in-Residual Dense Block
class RRDB(nn.Module):
    def __init__(self, channels=32, beta=0.2):
        super().__init__()
        self.beta = beta
        self.block1 = DenseBlock(channels)
        self.block2 = DenseBlock(channels)
        self.block3 = DenseBlock(channels)

    def forward(self, x):
        out = self.block1(x)
        out = self.block2(out)
        out = self.block3(out)
        return x + self.beta * out


# RD Block: Conv -> RRDBs -> Conv
class RDBlock(nn.Module):
    def __init__(self, in_channels, out_channels, num_rrdb=3):
        super().__init__()
        self.head = nn.Conv2d(in_channels, 128, kernel_size=3, padding=1)
        self.rrdb_blocks = nn.Sequential(*[RRDB(128) for _ in range(num_rrdb)])
        self.tail = nn.Conv2d(128, out_channels, kernel_size=3, padding=1)

    def forward(self, x):
        x = self.head(x)
        x = self.rrdb_blocks(x)
        x = self.tail(x)
        return x
    
# TODO check the input image sizes, bicubic not needed inside the model    
class Sen2RDSR(nn.Module):
    def __init__(self):
        super().__init__()
        self.sr20_block = RDBlock(in_channels=4 + 6, out_channels=6)  # 4m + upsampled 20m
        self.sr60_block = RDBlock(in_channels=4 + 6 + 2, out_channels=2)  # 4m + sr20 + upsampled 60m

        # Flags to control training stages
        self.train_sr20 = True
        self.train_sr60 = True

    def forward(self, im10, im20, im60):
        # TODO: check if bicubic is needed within the model
        #im20_up = F.interpolate(im20, scale_factor=2, mode='bicubic', align_corners=False)
        #im60_up = F.interpolate(im60, scale_factor=6, mode='bicubic', align_corners=False)
        im20_up = im20
        im60_up = im60
        
        if self.train_sr20:
            sr20_input = torch.cat([im10, im20_up], dim=1)
            sr20_residual = self.sr20_block(sr20_input)
            sr20 = im20_up + sr20_residual
        else:
            with torch.no_grad():
                sr20_input = torch.cat([im10, im20_up], dim=1)
                sr20_residual = self.sr20_block(sr20_input)
                sr20 = im20_up + sr20_residual

        if self.train_sr60:
            sr60_input = torch.cat([im10, sr20, im60_up], dim=1)
            sr60_residual = self.sr60_block(sr60_input)
            sr60 = im60_up + sr60_residual
        else:
            with torch.no_grad():
                sr60_input = torch.cat([im10, sr20, im60_up], dim=1)
                sr60_residual = self.sr60_block(sr60_input)
                sr60 = im60_up + sr60_residual

        return sr20, sr60

    def freeze_sr20(self):
        self.train_sr20 = False
        for p in self.sr20_block.parameters():
            p.requires_grad = False

    def freeze_sr60(self):
        self.train_sr60 = False
        for p in self.sr60_block.parameters():
            p.requires_grad = False

    def unfreeze_sr20(self):
        self.train_sr20 = True
        for p in self.sr20_block.parameters():
            p.requires_grad = True

    def unfreeze_sr60(self):
        self.train_sr60 = True
        for p in self.sr60_block.parameters():
            p.requires_grad = True

""" 
# Full Sen2-RDSR Model
class Sen2RDSR(nn.Module):
    def __init__(self):
        super().__init__()
        # Assume channel counts: 10m=4, 20m=6, 60m=2
        self.sr20_block = RDBlock(in_channels=10 + 6, out_channels=6)  # 10m + upsampled 20m
        self.sr60_block = RDBlock(in_channels=10 + 6 + 2, out_channels=2)  # 10m + sr20 + upsampled 60m

    def forward(self, im10, im20, im60):
        # Upsample LR inputs
        im20_up = F.interpolate(im20, scale_factor=2, mode='bicubic', align_corners=False)
        im60_up = F.interpolate(im60, scale_factor=6, mode='bicubic', align_corners=False)

        # SR20 branch
        sr20_input = torch.cat([im10, im20_up], dim=1)
        sr20_residual = self.sr20_block(sr20_input)
        sr20 = im20_up + sr20_residual

        # SR60 branch
        sr60_input = torch.cat([im10, sr20, im60_up], dim=1)
        sr60_residual = self.sr60_block(sr60_input)
        sr60 = im60_up + sr60_residual

        return sr20, sr60
"""