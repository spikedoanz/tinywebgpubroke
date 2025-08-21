import numpy
import nibabel as nib

webgpu_nifti_path, metal_nifti_path = "webgpu.nii.gz", "metal.nii.gz" 

webgpu_img  = nib.load(webgpu_nifti_path).get_fdata()
metal_img   = nib.load(metal_nifti_path).get_fdata()

l2 = (((webgpu_img - metal_img) ** 2) ** 0.5).sum()

print("sum l2 difference between two images", l2)
