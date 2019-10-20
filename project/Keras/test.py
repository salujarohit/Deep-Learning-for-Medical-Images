import matplotlib.pyplot as plt
from nibabel.testing import data_path
import os
import numpy as np
import nibabel as nib
from pathlib import Path
from Keras.visualize import visualize

visualize("00000","/test_interpolated")
# data_path = '/project_data/'
# # img = nib.Nifti1Image.from_filename(os.path.join(data_path, 'imaging.nii.gz'))
# img = nib.load(os.path.join(data_path, 'imaging_00000.nii.gz'))
# mask = nib.load(os.path.join(data_path, 'segmentation_00000.nii.gz'))
# fig, ax = plt.subplots(1, 2, figsize=(14, 2))
# ax[0].imshow(img,cmap="gray")
# ax[1].imshow(mask, cmap="gray")
# plt.show()
