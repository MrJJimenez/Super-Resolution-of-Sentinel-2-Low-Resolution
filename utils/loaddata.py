import numpy as np
import torch
import os

# Real Dataset for Sentinel-2 20m Super-Resolution
def normalize_band(band_data):
    """Normalize image data using percentile clipping"""
    if band_data is None or band_data.size == 0:
        return band_data
    
    # Clip to 1st and 99th percentiles to handle outliers
    p1, p99 = np.percentile(band_data, (1, 99))
    normalized = np.clip(band_data, p1, p99)
    
    # Normalize to [0, 1]
    if p99 > p1:
        normalized = (normalized - p1) / (p99 - p1)
    
    return normalized

def normalize_image_per_band(image):
    """Normalize each band of an image independently using percentile clipping."""
    # image: [C, H, W]
    normed = np.empty_like(image, dtype=np.float32)
    for c in range(image.shape[0]):
        normed[c] = normalize_band(image[c])
    return normed

class Sentinel2Dataset20IMG():
    def __init__(self, train_folder="train"):
        super().__init__()
        self.train_folder = train_folder

        # Find all subfolders containing the required .npy files
        
        self.samples = []
        for root, dirs, files in os.walk(self.train_folder):
            if (
                "data10.npy" in files
                and "data20.npy" in files
                and "data20_gt.npy" in files
            ):
                self.samples.append(root)

        # Load all .npy files and stack them
        self.im10 = []
        self.im20 = []
        self.target20 = []
        for folder in self.samples:
            im10_path = os.path.join(folder, "data10.npy")
            im20_path = os.path.join(folder, "data20.npy")
            target20_path = os.path.join(folder, "data20_gt.npy")
            im10_arr = np.load(im10_path)  # [N, 4, H, W]
            im20_arr = np.load(im20_path)  # [N, 6, H, W]
            target20_arr = np.load(target20_path)  # [N, 6, H, W]
            self.im10.append(im10_arr)
            self.im20.append(im20_arr)
            self.target20.append(target20_arr)
        self.im10 = np.concatenate(self.im10, axis=0)
        self.im20 = np.concatenate(self.im20, axis=0)
        self.target20 = np.concatenate(self.target20, axis=0)

    def __len__(self):
        return self.im10.shape[0]

    def __getitem__(self, idx):
        im10 = self.im10[idx]         # [4, H, W]
        im20 = self.im20[idx]         # [6, H/2, W/2]
        target20 = self.target20[idx] # [6, H, W]

        # Normalize each band of each image
        im10 = normalize_image_per_band(im10)
        im20 = normalize_image_per_band(im20)
        target20 = normalize_image_per_band(target20)

        # Convert to torch tensors
        im10 = torch.from_numpy(im10).float()
        im20 = torch.from_numpy(im20).float()
        target20 = torch.from_numpy(target20).float()
        return im10, im20, target20