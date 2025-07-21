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

class Sentinel2Dataset():
    def __init__(self, dataset_type="20train"):
        super().__init__()
        
        # Determine the folder based on dataset type
        if dataset_type == "20train":
            self.train_folder = "../data/train"
            self.load_60m = False
        elif dataset_type == "60train":
            self.train_folder = "../data/train60"
            self.load_60m = True
        else:
            raise ValueError("dataset_type must be either '20train' or '60train'")
        
        self.dataset_type = dataset_type
        print(f"Loading {dataset_type} data from {self.train_folder}")
        
        # Find all subfolders containing the required .npy files
        self.samples = []
        for root, dirs, files in os.walk(self.train_folder):
            if self.load_60m:
                # For 60train: require data10.npy, data20.npy, data60.npy, data60_gt.npy (no data20_gt.npy)
                required_files = ["data10.npy", "data20.npy", "data60.npy", "data60_gt.npy"]
            else:
                # For 20train: require data10.npy, data20.npy, data20_gt.npy
                required_files = ["data10.npy", "data20.npy", "data20_gt.npy"]
            
            if all(f in files for f in required_files):
                self.samples.append(root)
        
        print(f"Found {len(self.samples)} samples")
        
        # Load all .npy files and stack them
        self.im10 = []
        self.im20 = []
        self.target20 = []
        self.im60 = []
        self.target60 = []
        
        for folder in self.samples:
            im10_path = os.path.join(folder, "data10.npy")
            im20_path = os.path.join(folder, "data20.npy")
            
            im10_arr = np.load(im10_path)  # [N, 4, H, W]
            im20_arr = np.load(im20_path)  # [N, 6, H, W]
            
            self.im10.append(im10_arr)
            self.im20.append(im20_arr)
            
            if self.load_60m:
                # For 60train: load 60m data but target20 will be empty
                im60_path = os.path.join(folder, "data60.npy")
                target60_path = os.path.join(folder, "data60_gt.npy")
                im60_arr = np.load(im60_path)  # [N, 6, H, W]
                target60_arr = np.load(target60_path)  # [N, 6, H, W]
                self.im60.append(im60_arr)
                self.target60.append(target60_arr)
            else:
                # For 20train: load target20 data
                target20_path = os.path.join(folder, "data20_gt.npy")
                target20_arr = np.load(target20_path)  # [N, 6, H, W]
                self.target20.append(target20_arr)
        
        self.im10 = np.concatenate(self.im10, axis=0)
        self.im20 = np.concatenate(self.im20, axis=0)
        
        if self.load_60m:
            self.im60 = np.concatenate(self.im60, axis=0)
            self.target60 = np.concatenate(self.target60, axis=0)
        else:
            self.target20 = np.concatenate(self.target20, axis=0)

        # Normalize all images once during loading
        print("Normalizing images...")
        for i in range(len(self.im10)):
            self.im10[i] = normalize_image_per_band(self.im10[i])
            self.im20[i] = normalize_image_per_band(self.im20[i])
            
            if self.load_60m:
                self.im60[i] = normalize_image_per_band(self.im60[i])
                self.target60[i] = normalize_image_per_band(self.target60[i])
            else:
                self.target20[i] = normalize_image_per_band(self.target20[i])
        
        print("Normalization complete.")

    def __len__(self):
        return self.im10.shape[0]

    def __getitem__(self, idx):
        im10 = self.im10[idx]         # [4, H, W] - already normalized
        im20 = self.im20[idx]         # [6, H, W] - already normalized

        # Convert to torch tensors
        im10 = torch.from_numpy(im10).float()
        im20 = torch.from_numpy(im20).float()
        
        if self.load_60m:
            # For 60train: target20 is empty, load 60m data
            target20 = torch.empty(0)
            im60 = self.im60[idx]         # [6, H, W] - already normalized
            target60 = self.target60[idx] # [6, H, W] - already normalized
            im60 = torch.from_numpy(im60).float()
            target60 = torch.from_numpy(target60).float()
        else:
            # For 20train: load target20, 60m data is empty
            target20 = self.target20[idx] # [6, H, W] - already normalized
            target20 = torch.from_numpy(target20).float()
            im60 = torch.empty(0)
            target60 = torch.empty(0)
        
        return im10, im20, im60, target20, target60