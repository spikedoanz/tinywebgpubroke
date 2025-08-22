#!/usr/bin/env python3
"""
Detailed comparison of Conv-GroupNorm outputs between backends.
Focuses on finding exact locations where values diverge.
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

print("=" * 60)
print("DETAILED CONV→GROUPNORM COMPARISON: METAL vs WEBGPU")
print("=" * 60)

# Load data
metal_data = load_safetensor("conv_groupnorm_debug_metal.safetensors")
webgpu_data = load_safetensor("conv_groupnorm_debug_webgpu.safetensors")

# Focus on the main pipeline outputs
keys_to_check = ["conv_output", "groupnorm_output", "relu_output"]

for key in keys_to_check:
    print(f"\n{'='*60}")
    print(f"Analyzing: {key}")
    print("-" * 60)
    
    metal = metal_data[key]
    webgpu = webgpu_data[key]
    
    # Basic stats
    print(f"\nBasic statistics:")
    print(f"  Metal  - Min: {metal.min():.6f}, Max: {metal.max():.6f}, Mean: {metal.mean():.9f}")
    print(f"  WebGPU - Min: {webgpu.min():.6f}, Max: {webgpu.max():.6f}, Mean: {webgpu.mean():.9f}")
    
    # Compute differences
    diff = metal - webgpu
    abs_diff = np.abs(diff)
    
    print(f"\nDifference analysis:")
    print(f"  Max absolute diff: {abs_diff.max():.9f}")
    print(f"  Mean absolute diff: {abs_diff.mean():.9f}")
    print(f"  Std of diff: {diff.std():.9f}")
    
    # Find problematic locations
    threshold = 1.0
    problem_mask = abs_diff > threshold
    num_problems = problem_mask.sum()
    
    if num_problems > 0:
        print(f"\n⚠️  Found {num_problems} locations with |diff| > {threshold}")
        
        # Show first few problem locations
        problem_indices = np.where(problem_mask)
        num_to_show = min(5, num_problems)
        
        print(f"\nFirst {num_to_show} problematic locations:")
        for i in range(num_to_show):
            idx = tuple(p[i] for p in problem_indices)
            print(f"  Location {idx}:")
            print(f"    Metal:  {metal[idx]:.6f}")
            print(f"    WebGPU: {webgpu[idx]:.6f}")
            print(f"    Diff:   {diff[idx]:.6f}")
    
    # Special analysis for relu_output (where zeros matter)
    if key == "relu_output":
        print(f"\nZero analysis:")
        metal_zeros = (metal == 0).sum()
        webgpu_zeros = (webgpu == 0).sum()
        print(f"  Metal zero count:  {metal_zeros}")
        print(f"  WebGPU zero count: {webgpu_zeros}")
        print(f"  Difference: {metal_zeros - webgpu_zeros}")
        
        # Check if zeros are in different locations
        metal_zero_mask = (metal == 0)
        webgpu_zero_mask = (webgpu == 0)
        
        # Where Metal has zero but WebGPU doesn't
        metal_only_zeros = metal_zero_mask & ~webgpu_zero_mask
        # Where WebGPU has zero but Metal doesn't
        webgpu_only_zeros = webgpu_zero_mask & ~metal_zero_mask
        
        print(f"\n  Zeros only in Metal:  {metal_only_zeros.sum()}")
        print(f"  Zeros only in WebGPU: {webgpu_only_zeros.sum()}")
        
        if webgpu_only_zeros.sum() > 0:
            # Find cases where WebGPU incorrectly has 0
            indices = np.where(webgpu_only_zeros)
            num_to_show = min(5, webgpu_only_zeros.sum())
            print(f"\n  Examples where WebGPU has 0 but Metal doesn't:")
            for i in range(num_to_show):
                idx = tuple(p[i] for p in indices)
                print(f"    Location {idx}: Metal={metal[idx]:.6f}, WebGPU={webgpu[idx]:.6f}")

# Check Conv weights (they should be identical since we use the same seed)
print(f"\n{'='*60}")
print("Checking if issue starts at Conv weights...")
print("-" * 60)

# The weights are embedded in the model, so let's check the Conv output directly
# If Conv outputs differ significantly, the issue is in Conv computation
conv_metal = metal_data["conv_output"]
conv_webgpu = webgpu_data["conv_output"]

conv_diff = np.abs(conv_metal - conv_webgpu)
print(f"Conv output max difference: {conv_diff.max():.9f}")
print(f"Conv output mean difference: {conv_diff.mean():.9f}")

# Check correlation
correlation = np.corrcoef(conv_metal.flatten(), conv_webgpu.flatten())[0, 1]
print(f"Conv output correlation: {correlation:.9f}")

if conv_diff.max() > 0.1:
    print("\n⚠️  ISSUE FOUND: Conv outputs are already significantly different!")
    print("The problem is in the Conv operation, not GroupNorm!")
    
    # Find where Conv outputs differ most
    max_diff_idx = np.unravel_index(conv_diff.argmax(), conv_diff.shape)
    print(f"\nLocation of maximum Conv difference: {max_diff_idx}")
    print(f"  Metal Conv:  {conv_metal[max_diff_idx]:.6f}")
    print(f"  WebGPU Conv: {conv_webgpu[max_diff_idx]:.6f}")
    print(f"  Difference:  {conv_diff[max_diff_idx]:.6f}")

print("\n" + "=" * 60)
print("SUMMARY")
print("=" * 60)

# Calculate where the divergence amplifies
conv_max_diff = conv_diff.max()
gn_diff = np.abs(metal_data["groupnorm_output"] - webgpu_data["groupnorm_output"]).max()
relu_diff = np.abs(metal_data["relu_output"] - webgpu_data["relu_output"]).max()

print(f"\nMax absolute differences through the pipeline:")
print(f"  Conv output:      {conv_max_diff:.9f}")
print(f"  GroupNorm output: {gn_diff:.9f}")
print(f"  ReLU output:      {relu_diff:.9f}")

amplification_factor = relu_diff / conv_max_diff if conv_max_diff > 0 else 0
print(f"\nError amplification factor: {amplification_factor:.2f}x")

if conv_max_diff > 1.0:
    print("\n🔴 Root cause: Conv operation produces different results between backends")
elif gn_diff > conv_max_diff * 2:
    print("\n🔴 Root cause: GroupNorm amplifies small Conv differences")
else:
    print("\n🟡 Differences accumulate gradually through the pipeline")
