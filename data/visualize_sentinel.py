#!/usr/bin/env python
"""
Sentinel-2 Data Visualization Script
====================================

This script provides command-line tools to visualize Sentinel-2 satellite imagery 
data stored in .mat files using matplotlib.

Usage:
    python visualize_sentinel.py [filename.mat]
    
If no filename is provided, it will list available .mat files.

Requirements:
- numpy
- matplotlib  
- h5py (for MATLAB v7.3 files)
- scipy (for older MATLAB files)
"""

import numpy as np
import matplotlib.pyplot as plt
import h5py
import scipy.io as sio
import os
import sys
import glob
from pathlib import Path

def load_mat_file(filename):
    """
    Load .mat file using appropriate method (h5py for v7.3, scipy.io for older versions)
    """
    try:
        # First try with h5py (for MATLAB v7.3 files)
        with h5py.File(filename, 'r') as f:
            print(f"Loading {filename} using h5py (MATLAB v7.3 format)")
            data_dict = {}
            
            def extract_datasets(name, obj):
                if isinstance(obj, h5py.Dataset):
                    # Load the dataset and transpose if needed (MATLAB vs Python array ordering)
                    data = obj[()]
                    if data.ndim > 1:
                        data = np.transpose(data)
                    data_dict[name] = data
            
            # Visit all datasets in the file
            f.visititems(extract_datasets)
            return data_dict, 'h5py'
            
    except (OSError, KeyError, ImportError):
        try:
            # Fallback to scipy.io for older .mat files
            print(f"Loading {filename} using scipy.io (older MATLAB format)")
            data = sio.loadmat(filename)
            return data, 'scipy'
        except Exception as e:
            print(f"Error loading {filename} with both methods: {e}")
            return None, None

def load_sentinel_data(filename):
    """Load Sentinel-2 data from .mat file (supports both v7.3 and older formats)"""
    try:
        data, loader_type = load_mat_file(filename)
        if data is None:
            return None, None
            
        # Get the main data variable (usually the first non-metadata key)
        data_keys = [key for key in data.keys() if not key.startswith('__')]
        
        if data_keys:
            # Find the largest dataset (likely the main satellite image data)
            main_key = None
            max_size = 0
            
            for key in data_keys:
                if hasattr(data[key], 'size') and data[key].size > max_size:
                    max_size = data[key].size
                    main_key = key
            
            if main_key:
                main_data = data[main_key]
                print(f"Selected variable '{main_key}' as main data")
                print(f"Data shape: {main_data.shape}")
                print(f"Data type: {main_data.dtype}")
                
                # Handle different data structures
                if main_data.ndim == 2:
                    print("Data appears to be a 2D single-band image")
                elif main_data.ndim == 3:
                    print(f"Data appears to be a 3D multi-band image with {main_data.shape[2]} bands")
                else:
                    print(f"Data has {main_data.ndim} dimensions")
                
                return main_data, main_key
            else:
                print("No suitable data variable found")
                return None, None
        else:
            print("No data variables found in file")
            return None, None
            
    except Exception as e:
        print(f"Error loading {filename}: {e}")
        return None, None

def normalize_band(band_data, percentile_clip=2):
    """Normalize band data for visualization"""
    # Remove any invalid values
    valid_data = band_data[~np.isnan(band_data) & ~np.isinf(band_data)]
    
    if len(valid_data) == 0:
        return band_data
    
    # Clip outliers using percentiles
    p_low = np.percentile(valid_data, percentile_clip)
    p_high = np.percentile(valid_data, 100 - percentile_clip)
    
    # Normalize to 0-1 range
    normalized = np.clip((band_data - p_low) / (p_high - p_low), 0, 1)
    return normalized

def visualize_sentinel_image(data, variable_name, filename):
    """Create comprehensive visualization of Sentinel-2 data"""
    
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    fig.suptitle(f'Sentinel-2 Visualization: {filename}\nVariable: {variable_name}', 
                 fontsize=16, fontweight='bold')
    
    # If data has multiple bands (3D array)
    if len(data.shape) == 3:
        height, width, bands = data.shape
        
        # Plot 1: First band (grayscale)
        band1 = normalize_band(data[:, :, 0])
        im1 = axes[0, 0].imshow(band1, cmap='gray')
        axes[0, 0].set_title(f'Band 1 (Shape: {band1.shape})')
        axes[0, 0].axis('off')
        plt.colorbar(im1, ax=axes[0, 0], fraction=0.046, pad=0.04)
        
        # Plot 2: RGB composite (if at least 3 bands)
        if bands >= 3:
            rgb = np.zeros((height, width, 3))
            rgb[:, :, 0] = normalize_band(data[:, :, min(2, bands-1)])  # Red
            rgb[:, :, 1] = normalize_band(data[:, :, min(1, bands-1)])  # Green  
            rgb[:, :, 2] = normalize_band(data[:, :, 0])                # Blue
            
            axes[0, 1].imshow(rgb)
            axes[0, 1].set_title('RGB Composite')
            axes[0, 1].axis('off')
        else:
            axes[0, 1].text(0.5, 0.5, 'Not enough bands\nfor RGB composite', 
                           ha='center', va='center', transform=axes[0, 1].transAxes)
            axes[0, 1].axis('off')
        
        # Plot 3: Last band
        if bands > 1:
            band_last = normalize_band(data[:, :, -1])
            im3 = axes[0, 2].imshow(band_last, cmap='viridis')
            axes[0, 2].set_title(f'Band {bands} (Shape: {band_last.shape})')
            axes[0, 2].axis('off')
            plt.colorbar(im3, ax=axes[0, 2], fraction=0.046, pad=0.04)
        else:
            axes[0, 2].axis('off')
        
        # Plot 4: Statistics histogram
        axes[1, 0].hist(data.flatten(), bins=50, alpha=0.7, edgecolor='black')
        axes[1, 0].set_title('Pixel Value Distribution')
        axes[1, 0].set_xlabel('Pixel Value')
        axes[1, 0].set_ylabel('Frequency')
        axes[1, 0].grid(True, alpha=0.3)
        
        # Plot 5: Band comparison (if multiple bands)
        if bands > 1:
            band_means = [np.nanmean(data[:, :, i]) for i in range(min(bands, 10))]
            band_stds = [np.nanstd(data[:, :, i]) for i in range(min(bands, 10))]
            
            x_pos = range(len(band_means))
            axes[1, 1].bar(x_pos, band_means, yerr=band_stds, capsize=5, alpha=0.7)
            axes[1, 1].set_title('Band Statistics (Mean ± Std)')
            axes[1, 1].set_xlabel('Band Number')
            axes[1, 1].set_ylabel('Mean Value')
            axes[1, 1].grid(True, alpha=0.3)
        else:
            axes[1, 1].axis('off')
        
        # Plot 6: Data info
        info_text = f"""Data Information:
Shape: {data.shape}
Bands: {bands}
Data type: {data.dtype}
Min value: {np.nanmin(data):.2f}
Max value: {np.nanmax(data):.2f}
Mean value: {np.nanmean(data):.2f}
Std value: {np.nanstd(data):.2f}
Valid pixels: {np.sum(~np.isnan(data))}
NaN pixels: {np.sum(np.isnan(data))}"""
        
        axes[1, 2].text(0.05, 0.95, info_text, transform=axes[1, 2].transAxes, 
                        fontsize=10, verticalalignment='top', fontfamily='monospace')
        axes[1, 2].axis('off')
        
    # If data is 2D (single band)
    elif len(data.shape) == 2:
        band_norm = normalize_band(data)
        
        colormaps = ['gray', 'viridis', 'plasma', 'inferno']
        for i, cmap in enumerate(colormaps[:4]):
            row, col = i // 2, i % 2
            im = axes[row, col].imshow(band_norm, cmap=cmap)
            axes[row, col].set_title(f'Colormap: {cmap}')
            axes[row, col].axis('off')
            plt.colorbar(im, ax=axes[row, col], fraction=0.046, pad=0.04)
        
        # Statistics and info in remaining plots
        axes[1, 0].hist(data.flatten(), bins=50, alpha=0.7, edgecolor='black')
        axes[1, 0].set_title('Pixel Value Distribution')
        axes[1, 0].set_xlabel('Pixel Value')
        axes[1, 0].set_ylabel('Frequency')
        axes[1, 0].grid(True, alpha=0.3)
        
        info_text = f"""Data Information:
Shape: {data.shape}
Data type: {data.dtype}
Min value: {np.nanmin(data):.2f}
Max value: {np.nanmax(data):.2f}
Mean value: {np.nanmean(data):.2f}
Std value: {np.nanstd(data):.2f}
Valid pixels: {np.sum(~np.isnan(data))}
NaN pixels: {np.sum(np.isnan(data))}"""
        
        axes[1, 1].text(0.05, 0.95, info_text, transform=axes[1, 1].transAxes, 
                        fontsize=10, verticalalignment='top', fontfamily='monospace')
        axes[1, 1].axis('off')
    
    plt.tight_layout()
    plt.show()

def main():
    """Main function to handle command line arguments"""
    
    # Get all .mat files in current directory
    mat_files = glob.glob("*.mat")
    
    if len(sys.argv) < 2:
        print("Sentinel-2 Data Visualization Tool")
        print("="*40)
        
        if mat_files:
            print(f"Found {len(mat_files)} .mat files:")
            for i, file in enumerate(mat_files, 1):
                file_size = os.path.getsize(file) / (1024*1024)  # Size in MB
                print(f"  {i}. {file} ({file_size:.1f} MB)")
            
            print(f"\nUsage: python {sys.argv[0]} <filename.mat>")
            print("Example: python visualize_sentinel.py S2A_MSIL1C_20170527_T33UUB.mat")
            print("\nNote: This script supports both MATLAB v7.3 (HDF5) and older .mat formats")
        else:
            print("No .mat files found in current directory.")
        return
    
    filename = sys.argv[1]
    
    if not os.path.exists(filename):
        print(f"Error: File '{filename}' not found.")
        if mat_files:
            print("Available files:")
            for file in mat_files:
                print(f"  - {file}")
        return
    
    print(f"Visualizing {filename}...")
    data, variable_name = load_sentinel_data(filename)
    
    if data is not None:
        visualize_sentinel_image(data, variable_name, filename)
        print("Visualization complete!")
    else:
        print("Failed to load and visualize the data.")

if __name__ == "__main__":
    main() 