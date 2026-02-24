#!/usr/bin/env python3
"""
Test script to compare C++ T5Gemma2 image preprocessing with PyTorch reference.
"""

import numpy as np
import torch
from PIL import Image
import sys
import os

# Add the PyTorch model directory to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'res'))

# Import the PyTorch model
from transformers import Gemma3Processor, AutoModelForConditionalGeneration

def load_and_preprocess_image_pytorch(image_path, size=896):
    """
    Load and preprocess image using PyTorch transformers library.
    This matches the exact preprocessing done by T5Gemma2/Gemma3.
    """
    # Load the processor
    processor = Gemma3Processor.from_pretrained(os.path.join(os.path.dirname(__file__), 'res/t5gemma2'))
    
    # Load and process the image
    image = Image.open(image_path).convert('RGB')
    
    # Process the image
    inputs = processor(images=image, return_tensors="pt")
    
    # Get the pixel values (normalized)
    pixel_values = inputs['pixel_values'][0]  # Shape: (3, 896, 896)
    
    return pixel_values.numpy()

def save_txt(data, filename):
    """Save numpy array to text file."""
    np.savetxt(filename, data.flatten(), fmt='%.6f')
    print(f"Saved {data.size} values to {filename}")

def compare_arrays(cpp_data, pytorch_data, tolerance=1e-5):
    """Compare two numpy arrays and print statistics."""
    print(f"\n=== Comparison Results ===")
    print(f"C++ data shape: {cpp_data.shape}")
    print(f"PyTorch data shape: {pytorch_data.shape}")
    
    if cpp_data.shape != pytorch_data.shape:
        print("ERROR: Shape mismatch!")
        return False
    
    # Calculate statistics
    abs_diff = np.abs(cpp_data - pytorch_data)
    max_diff = np.max(abs_diff)
    mean_diff = np.mean(abs_diff)
    
    print(f"Max absolute difference: {max_diff:.10f}")
    print(f"Mean absolute difference: {mean_diff:.10f}")
    
    # Check if arrays are equal within tolerance
    if max_diff < tolerance:
        print(f"✓ Arrays match within tolerance ({tolerance})")
        return True
    else:
        print(f"✗ Arrays differ beyond tolerance")
        
        # Find and print some differing indices
        diff_indices = np.where(abs_diff > tolerance)
        num_diffs = len(diff_indices[0])
        print(f"Number of differing elements: {num_diffs}")
        
        if num_diffs > 0 and num_diffs <= 10:
            print("\nDifferences at first few indices:")
            for i in range(min(5, num_diffs)):
                idx = diff_indices[0][i]
                print(f"  Index {idx}: C++={cpp_data.flatten()[idx]:.6f}, PyTorch={pytorch_data.flatten()[idx]:.6f}, diff={abs_diff.flatten()[idx]:.6f}")
        
        return False

def main():
    # Get image path from command line or use default
    if len(sys.argv) > 1:
        image_path = sys.argv[1]
    else:
        image_path = os.path.join(os.path.dirname(__file__), 'res/t5gemma2/cat.png')
        print(f"No image path provided, using default: {image_path}")
    
    if not os.path.exists(image_path):
        print(f"ERROR: Image file not found: {image_path}")
        sys.exit(1)
    
    print(f"\n=== T5Gemma2 Image Preprocessing Comparison ===")
    print(f"Processing image: {image_path}\n")
    
    # Load PyTorch reference output
    print("Step 1: Loading PyTorch reference output...")
    pytorch_data = load_and_preprocess_image_pytorch(image_path)
    print(f"PyTorch data shape: {pytorch_data.shape}")
    print(f"PyTorch data range: [{np.min(pytorch_data):.6f}, {np.max(pytorch_data):.6f}]")
    print(f"PyTorch data mean: {np.mean(pytorch_data):.6f}")
    
    # Save PyTorch output
    save_txt(pytorch_data, 'pytorch_reference_output.txt')
    
    # Load C++ output (if it exists)
    cpp_output_file = 't5gemma2_preprocessed_output.txt'
    if os.path.exists(cpp_output_file):
        print(f"\nStep 2: Loading C++ output from {cpp_output_file}...")
        cpp_data = np.loadtxt(cpp_output_file)
        cpp_data = cpp_data.reshape(3, 896, 896)  # Reshape to CHW format
        
        print(f"C++ data shape: {cpp_data.shape}")
        print(f"C++ data range: [{np.min(cpp_data):.6f}, {np.max(cpp_data):.6f}]")
        print(f"C++ data mean: {np.mean(cpp_data):.6f}")
        
        # Compare
        match = compare_arrays(cpp_data, pytorch_data, tolerance=1e-5)
        
        if match:
            print("\n✓✓✓ SUCCESS: C++ implementation matches PyTorch reference!")
            sys.exit(0)
        else:
            print("\n✗✗✗ FAILURE: C++ implementation differs from PyTorch reference")
            sys.exit(1)
    else:
        print(f"\nStep 2: C++ output file not found: {cpp_output_file}")
        print("Please run the C++ test program first:")
        print("  cd Applications/CausalLM/models/t5gemma2")
        print("  ./test_image_preprocess [image_path]")
        sys.exit(1)

if __name__ == '__main__':
    main()
