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

class Sentinel2DatasetTest():
    def __init__(self, dataset_type="test60", base_data_path=None):
        super().__init__()
        
        # Auto-detect environment and set appropriate base path
        if base_data_path is None:
            try:
                # Check if running in Google Colab
                if 'google.colab' in str(get_ipython()):
                    # Running in Google Colab
                    base_data_path = "/content/drive/MyDrive/Colab Notebooks/Super-Resolution of Sentinel-2 Low Resolution/data"
                    print("🔍 Detected Google Colab environment")
                else:
                    # Running locally
                    base_data_path = "../data"
                    print("🔍 Detected local environment")
            except NameError:
                # get_ipython() not available (running in script), assume local
                base_data_path = "../data"
                print("🔍 Detected script environment (assuming local)")
        
        # Determine the folder based on dataset type
        if dataset_type == "test60":
            self.train_folder = os.path.join(base_data_path, "test60")
        else:
            raise ValueError("dataset_type must be 'test60'")
        
        self.dataset_type = dataset_type
        print(f"Loading {dataset_type} data from {self.train_folder}")
        
        # Check if the data folder exists
        if not os.path.exists(self.train_folder):
            print(f"❌ ERROR: Data folder not found at {self.train_folder}")
            try:
                if 'google.colab' in str(get_ipython()):
                    print("💡 COLAB SETUP INSTRUCTIONS:")
                    print("   1. Make sure you've mounted Google Drive")
                    print("   2. Upload your data folder to Google Drive at:")
                    print("      MyDrive/Colab Notebooks/Super-Resolution of Sentinel-2 Low Resolution/data/")
                    print("   3. Or specify a custom base_data_path parameter")
            except NameError:
                pass  # Not in notebook environment
            raise FileNotFoundError(f"Data folder not found: {self.train_folder}")
        
        # Find all subfolders containing the required .npy files
        self.samples = []
        for root, dirs, files in os.walk(self.train_folder):
            # Only require data10.npy, data20.npy, data60.npy (no target files)
            required_files = ["data10.npy", "data20.npy", "data60.npy"]
            
            if all(f in files for f in required_files):
                self.samples.append(root)
        
        print(f"Found {len(self.samples)} samples")
        
        # Load all .npy files and stack them
        self.im10 = []
        self.im20 = []
        self.im60 = []
        
        for folder in self.samples:
            im10_path = os.path.join(folder, "data10.npy")
            im20_path = os.path.join(folder, "data20.npy")
            im60_path = os.path.join(folder, "data60.npy")
            
            im10_arr = np.load(im10_path)  # [N, 4, H, W]
            im20_arr = np.load(im20_path)  # [N, 6, H, W]
            im60_arr = np.load(im60_path)  # [N, 2, H, W]
            
            self.im10.append(im10_arr)
            self.im20.append(im20_arr)
            self.im60.append(im60_arr)
        
        self.im10 = np.concatenate(self.im10, axis=0)
        self.im20 = np.concatenate(self.im20, axis=0)
        self.im60 = np.concatenate(self.im60, axis=0)

        # Normalize all images once during loading
        print("Normalizing images...")
        for i in range(len(self.im10)):
            self.im10[i] = normalize_image_per_band(self.im10[i])
            self.im20[i] = normalize_image_per_band(self.im20[i])
            self.im60[i] = normalize_image_per_band(self.im60[i])
        
        print("Normalization complete.")

    def __len__(self):
        return self.im10.shape[0]

    def __getitem__(self, idx):
        im10 = self.im10[idx]         # [4, H, W] - already normalized
        im20 = self.im20[idx]         # [6, H, W] - already normalized
        im60 = self.im60[idx]         # [2, H, W] - already normalized

        # Convert to torch tensors
        im10 = torch.from_numpy(im10).float()
        im20 = torch.from_numpy(im20).float()
        im60 = torch.from_numpy(im60).float()
        
        return im10, im20, im60