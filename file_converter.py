# -*- coding: utf-8 -*-
"""
Created on Wed Nov  4 17:57:57 2026

@author: eriks
"""
import numpy as np
import scipy.io
from pathlib import Path

# Setup file paths based on your workspace layout
uploaded_dir = Path(__file__).resolve().parent  # Update this path if needed
mat_file_path = uploaded_dir / "101000.mat"
npy_file_path = uploaded_dir / "receivedSignals.npy"

print(f"Loading '{mat_file_path.name}' entirely into memory...")

# 1. Load the entire MAT file workspace layout into RAM at once
# 'squeeze_me=True' removes unneeded single-element dimensions
mat_data = scipy.io.loadmat(str(mat_file_path), squeeze_me=True)

# 2. Extract the massive array Y
# This will be instantly instantiated as a native NumPy complex64 array
numpy_array = mat_data['Y']
print(f"Array 'Y' extracted. Shape: {numpy_array.shape}, Type: {numpy_array.dtype}")

# 3. Save the NumPy array natively straight to disk
# This creates a compact, raw binary file optimized for Python pipelines
np.save(str(npy_file_path), numpy_array)

print(f"Successfully converted and saved to: {npy_file_path}")
