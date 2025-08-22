#!/usr/bin/env python3
import os, numpy as np
from tinygrad import Tensor, nn, Device

# Conv->GroupNorm produces different results on larger tensors

np.random.seed(42)

def test_conv_gn_relu(b,c1,c2,N):
  print("Channels", b,c1,c2, "Dims", N)
  data = np.random.randn(b, c1, N,N,N).astype(np.float32)
  weights = np.random.randn(c1, c2, 3, 3, 3).astype(np.float32)# * 0.1

  for backend in ["METAL", "WEBGPU"]:
    os.environ.pop('WEBGPU', None) if backend == "METAL" else os.environ.update({'WEBGPU': '1'})
    Device.DEFAULT = backend
    x = Tensor(data)
    conv = nn.Conv2d(c1, c2, kernel_size=(3, 3, 3), padding=1, bias=False)
    conv.weight.assign(Tensor(weights)).realize()
    gn = nn.GroupNorm(c2, c2, affine=False)
    
    out = gn(x).realize().numpy()
    print(f"{backend:6s}: Min={out.min():.3f}, Max={out.max():.3f}, Mean={out.mean():.3f}")
    if backend == "METAL": metal_out = out
    else: print(f"Max diff: {np.abs(metal_out - out).max():.3f}")

# by "breaks" i mean sizable diff. in this case > 1.0. typical diff is 5.0 

b,c1,c2,N = [1,30,30,128] # works 
test_conv_gn_relu(b,c1,c2,N)

print("sweep over input dimensions") # ==============
print("="*80)
#b,c1,c2,N = [1,30,30,128+32] # works on everything smaller
#test_conv_gn_relu(b,c1,c2,N)
b,c1,c2,N = [1,30,30,128+36] # breaks on everything bigger
test_conv_gn_relu(b,c1,c2,N)

print("sweep over inner channels") # ==============
print("="*80)
b,c1,c2,N = [1,5,5,128+36] # channels [5,10,15,30] break
test_conv_gn_relu(b,c1,c2,N)
