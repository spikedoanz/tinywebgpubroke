#!/bin/sh

WEBGPU=1 python tiny_meshnet.py
mv segmentation_output.nii.gz webgpu.nii.gz
METAL=1 python tiny_meshnet.py
mv segmentation_output.nii.gz metal.nii.gz

python diff.py
