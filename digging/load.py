from tinygrad import Tensor, nn
import nibabel as nib
import numpy as np


def get_norm(t:Tensor) -> float:
  norm = (t**2**0.5).sum().item()
  return norm

# loading nifti image seems fine
img = nib.load('t1_crop.nii.gz').get_fdata().astype(np.int32)
t = Tensor(img).rearrange("... -> 1 1 ...")
t = (t - 0.0).div(255.0)
t = t.clip(0,1)
print("img l2 norm", get_norm(t))

# trying conv kernel
#for i in range(4,17,4): # all pass
for i in range(4,5,4):
  k_sz = (i,i,i)
  k = nn.Conv2d(1, 15, k_sz)
  t1 = k(t)
  print(f"norm after kernel size {i} {get_norm(t1)}")


# is it channel count?
channels = [1,4,16,64,16,4,1]
_t = t.clone()
for c_in, c_out in zip(channels[:-1], channels[1:]):
  k_sz = (2,2,2)
  k = nn.Conv2d(c_in, c_out, k_sz)
  _t = k(_t)
  print(f"norm going from kernel sz {c_in} -> {c_out}: {get_norm(_t)}")
