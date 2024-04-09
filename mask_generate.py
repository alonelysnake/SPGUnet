import os
import shutil
from glob import glob
from tqdm import tqdm
import SimpleITK as sitk
import numpy as np

dir1 = '/root/autodl-tmp/nnUNet_raw/Dataset119_Vessel'
dir2 = '/root/autodl-fs/fold_4'
for c in os.listdir(dir2):
    idx = c.split('-')[0]
    if 'nii' not in c:
        continue
    vol = sitk.ReadImage(os.path.join(dir2, c))
    v = sitk.GetArrayFromImage(vol)
    v = np.where(v == 3, 1, 0)
    v = sitk.GetImageFromArray(v)
    v.SetOrigin(vol.GetOrigin())
    v.SetSpacing(vol.GetSpacing())
    v.SetDirection(vol.GetDirection())
    if int(idx) > 130:
        sitk.WriteImage(v, os.path.join(dir1, 'imagesTs', c.replace('.nii', '_0001.nii')))
    else:
        sitk.WriteImage(v, os.path.join(dir1, 'imagesTr', c.replace('.nii', '_0001.nii')))
exit(0)