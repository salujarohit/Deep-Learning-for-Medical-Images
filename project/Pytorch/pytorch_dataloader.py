import numpy as np
import os
from skimage.io import imread
from skimage.transform import resize
import random
from scipy.ndimage.morphology import binary_fill_holes
from sklearn.model_selection import train_test_split
import SimpleITK as sitk
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms.functional as TF
import torch
from PIL import Image

class MyDataset(Dataset):
    def __init__(self, image_names, mask_names, hyperparameters, step_num, fold_num, generator_type, transform=None):
        self.hyperparameters = hyperparameters
        self.image_filenames = image_names
        self.mask_filenames = mask_names
        self.step_num = step_num
        self.fold_num = fold_num
        self.generator_type = generator_type
        self.transform = transform

    def __len__(self):
        return int(np.ceil(len(self.image_filenames)))

    @staticmethod
    def to_binary(img, threshold):
        img[img < threshold] = 0
        img[img >= threshold] = 1
        return img

    @staticmethod
    def to_categorical(img, num_classes):
        shape = list(img.shape)
        shape.append(num_classes)
        mask = np.zeros(shape)
        for i, value in enumerate(np.unique(img)):
            mask[(img == value), i] = 1
        return mask

    def normalize(self,image):
        pass

    def get_x(self, idx):
        x = imread(self.image_filenames[idx], as_gray=False)
        x = np.array(resize(x, (self.hyperparameters['input_shape'][0], self.hyperparameters['input_shape'][0])))

        if self.hyperparameters['input_shape'][2] == 1:
            x = np.expand_dims(x, axis=3)

        return x

    def get_y(self, idx):
        mask = imread(self.mask_filenames[idx], as_gray=False)
        mask = resize(mask, (self.hyperparameters['input_shape'][0], self.hyperparameters['input_shape'][1]), order=0, anti_aliasing=False,
                          preserve_range=True)
        # if self.hyperparameters['input_shape'][2] == 1:
        #     mask = np.expand_dims(mask, axis=3)

        if self.hyperparameters['last_layer_units'] == 1:
            mask = self.to_binary(mask, self.hyperparameters['threshold'])
        else:
            mask = self.to_categorical(mask, num_classes=self.hyperparameters['last_layer_units'])

        y = np.array(mask)

        return y

    @staticmethod
    def get_mask_boundary(masks, radius_dil=2, radius_ero=2):
        mask_boundaries = np.zeros(masks.shape)
        for i, mask in enumerate(masks):
            mask_img = sitk.GetImageFromArray(mask, isVector=False)
            mask_dilated = sitk.GrayscaleDilate(mask_img, radius_dil)
            mask_dilated = binary_fill_holes(sitk.GetArrayFromImage(mask_dilated)).astype(mask.dtype)
            mask_eroded = sitk.GrayscaleErode(mask_img, radius_ero)
            mask_eroded = binary_fill_holes(sitk.GetArrayFromImage(mask_eroded)).astype(mask.dtype)
            mask_boundaries[i] = mask_dilated - mask_eroded
        return mask_boundaries

    def __transform__(self, x, y, z=None):

        # # Random crop
        # i, j, h, w = transforms.RandomCrop.get_params(
        #     x, output_size=(512, 512))
        # x = TF.crop(x, i, j, h, w)
        # y = TF.crop(y, i, j, h, w)
        #
        # # Random horizontal flipping
        # if random.random() > 0.5:
        #     x = TF.hflip(x)
        #     y = TF.hflip(y)
        #
        # # Random vertical flipping
        # if random.random() > 0.5:
        #     x = TF.vflip(x)
        #     y = TF.vflip(y)
        x = x.transpose((2, 0, 1))
        x = torch.from_numpy(x)
        y = y.transpose((2, 0, 1))
        y = torch.from_numpy(y)

        if z:
            z = z.transpose((2, 0, 1))
            z = torch.from_numpy(z)
            return x, y, z
        else:
            return x, y

    def __getitem__(self, idx):
        x = self.get_x(idx)
        y = self.get_y(idx)
        if self.hyperparameters.get('use_weight_maps'):
            mask_boundary = self.get_mask_boundary(y)
            if self.transform:
                x, y, mask_boundary = self.__transform__(x, y, mask_boundary)
            return [x, y, mask_boundary]
        else:
            if self.transform:
                x, y = self.__transform__(x, y)
            return [x, y]

class MyDataloader:
    def __init__(self, hyperparameters):
        self.hyperparameters = hyperparameters
        self.load_filenames()
        if self.hyperparameters.get('folds'):
            self.split_data_folds()
        else:
            self.split_data()

    @staticmethod
    def combine_generators(gen1, gen2, gen3=None):
        batch_x = gen1.next()
        batch_y = gen2.next()
        if gen3:
            batch_w = gen3.next()
            return [batch_x, batch_w], batch_y
        else:
            return batch_x, batch_y

    def load_filenames(self):
        image_path = os.path.join(os.getcwd(), os.path.join(self.hyperparameters['data_path'], 'Image'))
        mask_path = os.path.join(os.getcwd(), os.path.join(self.hyperparameters['data_path'], 'Mask'))
        image_names = np.array([os.path.join(image_path, file) for file in os.listdir(image_path)])
        mask_names = np.array([os.path.join(mask_path, file) for file in os.listdir(mask_path)])
        image_names.sort()
        mask_names.sort()

        if self.hyperparameters.get('shuffle'):
            paired = list(zip(image_names, mask_names))
            random.Random(4).shuffle(paired)
            separated = [list(t) for t in zip(*paired)]
            image_names = np.array(separated[0])
            mask_names = np.array(separated[1])
        self.image_names, self.mask_names = image_names, mask_names

    def split_data(self):
        self.x_train, self.x_val, self.y_train, self.y_val = train_test_split(self.image_names, self.mask_names,
                                                                              test_size=self.hyperparameters['test_size'],
                                                                              random_state=1)

    def split_data_folds(self):
        fold_len = len(self.image_names) // self.hyperparameters['folds']
        images_indices = list(np.arange(len(self.image_names)))
        self.folds = []
        for i in range(0, self.hyperparameters['folds']):
            validation_indices = list(np.arange(i * fold_len, (i + 1) * fold_len))
            training_indices = list(set(images_indices) - set(validation_indices))
            validation_images = self.image_names[validation_indices]
            validation_masks = self.mask_names[validation_indices]
            training_images = self.image_names[training_indices]
            training_masks = self.mask_names[training_indices]
            self.folds.append([training_images, training_masks, validation_images, validation_masks])

    def get_generators(self, fold_num=None, step_num=None):
        if self.hyperparameters.get('folds'):
            x_train, y_train, x_val, y_val = self.folds[fold_num]
        else:
            x_train, y_train, x_val, y_val = self.x_train, self.y_train, self.x_val, self.y_val

        train_data = MyDataset(x_train, y_train, self.hyperparameters, step_num=step_num,
                                               fold_num=fold_num, generator_type="training", transform=True)
        val_data = MyDataset(x_val, y_val, self.hyperparameters, step_num=step_num,
                                               fold_num=fold_num, generator_type="testing", transform=True)
        train_loader = DataLoader(train_data, batch_size=self.hyperparameters['batch_size'], num_workers=0)
        val_loader = DataLoader(val_data, batch_size=self.hyperparameters['batch_size'], num_workers=0)

        return train_loader, val_loader

