#!/usr/bin/env python3
"""
Compare GroupNorm outputs between Metal and WebGPU backends.
Run this after running groupnorm_test.py with both backends.
"""

import numpy as np
from safetensors import safe_open

def load_safetensor(filepath):
    """Load all tensors from a safetensors file"""
    tensors = {}
    with safe_open(filepath, framework="numpy") as f:
        for key in f.keys():
            tensors[key] = f.get_tensor(key)
    return tensors

def compare_tensors(name, metal_tensor, webgpu_tensor):
    """Compare two tensors and print statistics"""
    print(f"\n{name}:")
    print("-" * 40)
    
    # Basic stats for each
    print(f"Metal   - Min: {metal_tensor.min():.6f}, Max: {metal_tensor.max():.6f}, Mean: {metal_tensor.mean():.9f}")
    print(f"WebGPU  - Min: {webgpu_tensor.min():.6f}, Max: {webgpu_tensor.max():.6f}, Mean: {webgpu_tensor.mean():.9f}")
    
    # Difference stats
    diff = metal_tensor - webgpu_tensor
    abs_diff = np.abs(diff)
    
    print(f"\nDifference statistics:")
    print(f"  Max absolute diff: {abs_diff.max():.9f}")
    print(f"  Mean absolute diff: {abs_diff.mean():.9f}")
    print(f"  Relative error: {(abs_diff / (np.abs(metal_tensor) + 1e-10)).mean():.6%}")
    
    # Check for NaNs
    metal_nans = np.isnan(metal_tensor).sum()
    webgpu_nans = np.isnan(webgpu_tensor).sum()
    if metal_nans > 0 or webgpu_nans > 0:
        print(f"  ⚠️  NaN count - Metal: {metal_nans}, WebGPU: {webgpu_nans}")
    
    # Check if values are drastically different
    if abs_diff.max() > 1.0:
        print(f"  ⚠️  WARNING: Large difference detected (> 1.0)")
        # Find location of max difference
        max_idx = np.unravel_index(abs_diff.argmax(), abs_diff.shape)
        print(f"  Location of max diff: {max_idx}")
        print(f"  Metal value: {metal_tensor[max_idx]:.6f}")
        print(f"  WebGPU value: {webgpu_tensor[max_idx]:.6f}")
    
    return abs_diff.max(), abs_diff.mean()

# Main comparison
print("=" * 60)
print("COMPARING GROUPNORM OUTPUTS: METAL vs WEBGPU")
print("=" * 60)

try:
    metal_data = load_safetensor("groupnorm_metal.safetensors")
    print("✓ Loaded Metal outputs")
except FileNotFoundError:
    print("❌ groupnorm_metal.safetensors not found. Run: python groupnorm_test.py")
    exit(1)

try:
    webgpu_data = load_safetensor("groupnorm_webgpu.safetensors")
    print("✓ Loaded WebGPU outputs")
except FileNotFoundError:
    print("❌ groupnorm_webgpu.safetensors not found. Run: WEBGPU=1 python groupnorm_test.py")
    exit(1)

# Compare each tensor
max_diffs = {}
mean_diffs = {}

for key in metal_data.keys():
    if key in webgpu_data:
        max_diff, mean_diff = compare_tensors(key, metal_data[key], webgpu_data[key])
        max_diffs[key] = max_diff
        mean_diffs[key] = mean_diff
    else:
        print(f"⚠️  Key '{key}' not found in WebGPU data")

# Summary
print("\n" + "=" * 60)
print("SUMMARY")
print("=" * 60)

print("\nMaximum differences:")
for key, diff in sorted(max_diffs.items(), key=lambda x: x[1], reverse=True):
    status = "🔴" if diff > 1.0 else "🟡" if diff > 0.1 else "🟢"
    print(f"  {status} {key:30s}: {diff:.9f}")

print("\nMean absolute differences:")
for key, diff in sorted(mean_diffs.items(), key=lambda x: x[1], reverse=True):
    status = "🔴" if diff > 0.1 else "🟡" if diff > 0.01 else "🟢"
    print(f"  {status} {key:30s}: {diff:.9f}")

# Check if fusion makes a difference
if "conv_groupnorm_relu_nofusion" in max_diffs and "conv_groupnorm_relu_fusion" in max_diffs:
    fusion_impact = max_diffs["conv_groupnorm_relu_fusion"] - max_diffs["conv_groupnorm_relu_nofusion"]
    print(f"\nFusion impact on max difference: {fusion_impact:+.9f}")
    if abs(fusion_impact) > 0.01:
        print("  ⚠️  Fusion significantly affects numerical differences!")

print("\n" + "=" * 60)
print("Analysis complete!")
