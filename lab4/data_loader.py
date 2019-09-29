import numpy as np
import os
from skimage.io import imread
from skimage.color import rgb2gray
from skimage.transform import resize
import random
from scipy.ndimage.morphology import binary_fill_holes
from sklearn.model_selection import train_test_split
import SimpleITK as sitk
try:
    from tensorflow.keras.processing.image import ImageDataGenerator, array_to_img
    from tensorflow.keras.utils import Sequence
except:
    from tensorflow.python.keras.preprocessing.image import ImageDataGenerator, array_to_img
    from tensorflow.python.keras.utils import Sequence


def binarize(img, values=[255]):
    mask = np.zeros(img.shape)
    for value in values:
        mask[img == value] = value
    return mask


def unify(img, unify_dict):
    for key in unify_dict:
        img[img == int(key)] = int(unify_dict[key])
    return img


def to_categorical(img, num_classes):
    shape = list(img.shape)
    shape[2] = num_classes
    mask = np.zeros(shape)
    for i, value in enumerate(np.unique(img)):
        mask[(img == value)[:, :, 0], i] = 1
    return mask


def combine_generators(gen1, gen2, gen3=None):
    batch_x = gen1.next()
    batch_y = gen2.next()
    if gen3:
        batch_w = gen3.next()
        return [batch_x, batch_w], batch_y
    else:
        return batch_x, batch_y


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


class MyGenerator(Sequence):
    def __init__(self, image_filenames, mask_filenames, batch_size, input_shape, generator_features, hyperparameters):
        self.image_filenames, self.mask_filenames = image_filenames, mask_filenames
        self.batch_size = batch_size
        self.input_shape = input_shape
        self.generator_features = generator_features
        self.hyperparameters = hyperparameters

    def __len__(self):
        return int(np.ceil(len(self.image_filenames) / float(self.batch_size)))

    def __getitem__(self, idx):
        batch_x = self.image_filenames[idx * self.batch_size:(idx + 1) * self.batch_size]
        batch_y = self.mask_filenames[idx * self.batch_size:(idx + 1) * self.batch_size]
        data_gen = ImageDataGenerator(** self.generator_features)
        x = np.array(
            [resize(imread(file_name, as_gray=False), (self.input_shape[0], self.input_shape[1])) for file_name in
             batch_x])
        if self.input_shape[2] == 1:
            x = np.expand_dims(x, axis=3)
        y_list = []
        for file_name in batch_y:
            mask = resize(imread(file_name, as_gray=False), (self.input_shape[0], self.input_shape[1]))
            if self.input_shape[2] == 1:
                mask = np.expand_dims(mask, axis=3)
            if 'binarize_values' in self.hyperparameters:
                mask = binarize(mask, self.hyperparameters['binarize_values'])
            if "unify_dict" in self.hyperparameters:
                mask = unify(mask, self.hyperparameters['unify_dict'])
            if self.hyperparameters['last_layer_units'] > 1:
                mask = to_categorical(mask, num_classes=self.hyperparameters['last_layer_units'])
            y_list.append(mask)
        y = np.array(y_list)

        image_data_gen = data_gen.flow(x, batch_size=self.batch_size, seed=100)
        mask_data_gen = data_gen.flow(y, batch_size=self.batch_size, seed=100)

        mask_boundary_data_gen = None
        if 'use_weight_maps' in self.hyperparameters and self.hyperparameters['use_weight_maps']:
            mask_boundaries = get_mask_boundary(y)
            mask_boundary_data_gen = data_gen.flow(mask_boundaries, batch_size=self.batch_size, seed=100)

        return combine_generators(image_data_gen, mask_data_gen, mask_boundary_data_gen)
 

def get_data_with_generator_on_the_fly(hyperparameters, x_train, y_train, x_test, y_test):
    data_generator = hyperparameters['generator'] if 'generator' in hyperparameters else {}
    training_batch_generator = MyGenerator(x_train, y_train, hyperparameters['batch_size'],
                                           hyperparameters['input_shape'], data_generator, hyperparameters)
    data_generator = hyperparameters['test_generator'] if 'test_generator' in hyperparameters else {}
    validation_batch_generator = MyGenerator(x_test, y_test, hyperparameters['batch_size'],
                                             hyperparameters['input_shape'], data_generator, hyperparameters)
    return training_batch_generator, validation_batch_generator, len(y_train), len(y_test)


def split_data_to_folds(image_names, mask_names, num_folds, test_size=None, shuffle=True):
    if shuffle:
        paired = list(zip(image_names, mask_names))
        random.shuffle(paired)
        separated = [list(t) for t in zip(*paired)]
        image_names = separated[0]
        mask_names = separated[1]
    image_names = np.array(image_names)
    mask_names = np.array(mask_names)
    if num_folds == 1:
        training_images, validation_images, training_masks, validation_masks = train_test_split(image_names, mask_names,
                                                                                                test_size=test_size,
                                                                                                random_state=1)
        yield training_images, training_masks, validation_images, validation_masks
    else:
        num_data_per_fold = len(image_names) // num_folds
        images_indices = list(np.arange(len(image_names)))
        for i in range(0, num_folds):
            validation_indices = list(np.arange(i * num_data_per_fold, (i + 1) * num_data_per_fold))
            training_indices = list(set(images_indices) - set(validation_indices))
            validation_images = image_names[validation_indices]
            validation_masks = mask_names[validation_indices]
            training_images = image_names[training_indices]
            training_masks = mask_names[training_indices]
            yield training_images, training_masks, validation_images, validation_masks


def get_folds(hyperparameters):
    image_path = os.path.join(os.getcwd(), os.path.join(hyperparameters['data_path'], 'Image'))
    mask_path = os.path.join(os.getcwd(), os.path.join(hyperparameters['data_path'], 'Mask'))
    image_names = [os.path.join(image_path, file) for file in os.listdir(image_path)]
    mask_names = [os.path.join(mask_path, file) for file in os.listdir(mask_path)]
    image_names.sort()
    mask_names.sort()
    test_size = None if 'test_size' not in hyperparameters else hyperparameters['test_size']
    folds = split_data_to_folds(image_names, mask_names, hyperparameters['num_folds'], test_size=test_size)
    return folds

