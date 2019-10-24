from pathlib import Path
import scipy.misc
import numpy as np
import os
import nibabel as nib
from skimage.transform import resize

class PreProcessing:
    def __init__(self, hu_min=-512, hu_max=512, source=None, destination=None, resize_shape=(240, 240)):
        self.hu_min = hu_min
        self.hu_max = hu_max
        self.source = source
        self.destination = destination
        self.resize_shape = (resize_shape[0],resize_shape[1])

    def set_source(self, source):
        self.source = source

    def set_destination(self, destination):
        self.destination = destination

    def load_volume(self, case_id):
        case_id = str(case_id).zfill(5)
        img_name = ''
        for file in os.listdir(self.source):
            if str(case_id) in file and 'imaging' in file:
                img_name = file
        img = nib.load(os.path.join(self.source, img_name))
        return img

    def load_case(self, case_id):
        case_id = str(case_id).zfill(5)
        img_name = ''
        mask_name = ''
        for file in os.listdir(self.source):
            if str(case_id) in file and 'imaging' in file:
                img_name = file
            elif str(case_id) in file and 'segmentation' in file:
                mask_name = file
        img = nib.load(os.path.join(self.source, img_name))
        mask = nib.load(os.path.join(self.source, mask_name))
        return img, mask

    def hu_to_grayscale(self, volume):
        # Clip at max and min values if specified
        if self.hu_min is not None or self.hu_max is not None:
            volume = np.clip(volume, self.hu_min, self.hu_max)

        # Scale to values between 0 and 1
        mxval = np.max(volume)
        mnval = np.min(volume)
        im_volume = (volume - mnval) / max(mxval - mnval, 1e-3)

        # Return values scaled to 0-255 range, but *not cast to uint8*
        # Repeat three times to make compatible with color overlay
        im_volume = 255 * im_volume
        return im_volume

    @staticmethod
    def create_dir(dir):
        dir_path = Path(dir)
        if not dir_path.exists():
            dir_path.mkdir()
        return dir_path

    def preprocess (self, num_patients = 10, starting_patient=0):
        if self.source is None and self.destination is None:
            raise ValueError("Please make sure to set source and destination")
        in_path = Path(self.source)
        if not in_path.exists():
            raise ValueError("Source directory doesn't exist")

        _ = self.create_dir(self.destination)
        image_path = self.destination+'Image'
        image_path = self.create_dir(image_path)
        mask_path = self.destination+'Mask'
        mask_path = self.create_dir(mask_path)

        for i in range(starting_patient, starting_patient+num_patients):
            # Load segmentation and volume
            vol, seg = self.load_case(i)
            vol = vol.get_data()
            seg = seg.get_data()
            seg = seg.astype(np.int32)

            # Convert to a visual format
            vol_ims = self.hu_to_grayscale(vol)

            for j in range(vol_ims.shape[0]):
                #extracting only images that have tumor
                if np.max(seg[j]) != 0: #np.max(seg[j]) == 2:
                    file_path = image_path / ("{}_{:05d}.png".format(i, j))
                    image = resize(vol_ims[j], self.resize_shape)
                    scipy.misc.imsave(str(file_path), image)
                    file_path = mask_path / ("{}_{:05d}.png".format(i, j))
                    mask = resize(seg[j], self.resize_shape,
                           order=0, anti_aliasing=False,
                           preserve_range=True)
                    scipy.misc.imsave(str(file_path), mask)

    def preprocess_predictions(self,num_patients, starting_patient):
        if self.source is None and self.destination is None:
            raise ValueError("Please make sure to set source and destination")
        in_path = Path(self.source)
        if not in_path.exists():
            raise ValueError("Source directory doesn't exist")

        _ = self.create_dir(self.destination)
        image_path = self.destination+'test_images'
        image_path = self.create_dir(image_path)

        for i in range(starting_patient, starting_patient+num_patients):
            # Load segmentation and volume
            vol = self.load_volume(i)
            vol = vol.get_data()

            # Convert to a visual format
            vol_ims = self.hu_to_grayscale(vol)

            for j in range(vol_ims.shape[0]):
                file_path = image_path / ("{:05d}_{:03d}.png".format(i, j))
                scipy.misc.imsave(str(file_path), vol_ims[j])