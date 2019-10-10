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
from models import load_step_prediction
import pandas as pd
import nibabel as nib
from sklearn.preprocessing import MinMaxScaler


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
            mask = imread(file_name, as_gray=False)
            mask = resize(mask, (self.input_shape[0], self.input_shape[1]), order=0, anti_aliasing=False, preserve_range=True)
            if max(np.unique(mask)) == 255:
                mask /= 255
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

        return combine_generators(image_data_gen, mask_data_gen)
 

def get_data_with_generator_on_the_fly(hyperparameters):
    image_path = os.path.join(os.getcwd(), os.path.join(hyperparameters['data_path'], 'Image'))
    mask_path = os.path.join(os.getcwd(), os.path.join(hyperparameters['data_path'], 'Mask'))
    image_names = [os.path.join(image_path, file) for file in os.listdir(image_path)]
    mask_names = [os.path.join(mask_path, file) for file in os.listdir(mask_path)]
    image_names.sort()
    mask_names.sort()
    x_train, x_test, y_train, y_test = train_test_split(image_names, mask_names, test_size=hyperparameters['test_size'],
                                                        random_state=1)
    data_generator = hyperparameters['generator'] if 'generator' in hyperparameters else {}
    training_batch_generator = MyGenerator(x_train, y_train, hyperparameters['batch_size'],
                                           hyperparameters['input_shape'], data_generator, hyperparameters)
    data_generator = hyperparameters['test_generator'] if 'test_generator' in hyperparameters else {}
    validation_batch_generator = MyGenerator(x_test, y_test, hyperparameters['batch_size'],
                                             hyperparameters['input_shape'], data_generator, hyperparameters)
    return training_batch_generator, validation_batch_generator


def task1_loader(hyperparameters):
    # load csv files:
    dataset_train = pd.read_csv('/Lab1/Lab5/train_data_stock.csv')
    dataset_val = pd.read_csv('/Lab1/Lab5/val_data_stock.csv')
    # reverse data so that they go from oldest to newest:
    dataset_train = dataset_train.iloc[::-1]
    dataset_val = dataset_val.iloc[::-1]
    # concatenate training and test datasets:
    dataset_total = pd.concat((dataset_train['Open'], dataset_val['Open']), axis=0)
    # select the values from the “Open” column as the variables to be predicted:
    training_set = dataset_train.iloc[:, 1:2].values
    val_set = dataset_val.iloc[:, 1:2].values
    sc = MinMaxScaler(feature_range=(0, 1))
    training_set_scaled = sc.fit_transform(training_set)
    # split training data into T time steps:
    X_train = []
    y_train = []
    for i in range(hyperparameters['time_steps'], len(training_set)):
        X_train.append(training_set_scaled[i - hyperparameters['time_steps']:i, 0])
        y_train.append(training_set_scaled[i, 0])
    X_train, y_train = np.array(X_train), np.array(y_train)
    # normalize the validation set according to the normalization applied to the training set:
    inputs = dataset_total[len(dataset_total) - len(dataset_val) - 60:].values
    inputs = inputs.reshape(-1, 1)
    inputs = sc.transform(inputs)
    # split validation data into T time steps:
    X_val = []
    for i in range(hyperparameters['time_steps'], hyperparameters['time_steps'] + len(val_set)):
        X_val.append(inputs[i - hyperparameters['time_steps']:i, 0])
    X_val = np.array(X_val)
    y_val = sc.transform(val_set)
    # reshape to 3D array (format needed by LSTMs -> number of samples, timesteps, input dimension)
    X_train = np.reshape(X_train, (X_train.shape[0], X_train.shape[1], 1))
    X_val = np.reshape(X_val, (X_val.shape[0], X_val.shape[1], 1))
    return X_train, y_train, X_val, y_val


class MyBatchGenerator(Sequence):
    def __init__(self, X, y, batch_size=1, shuffle=True):
        self.X = X
        self.y = y
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.on_epoch_end()

    def __len__(self):
        return int(np.floor(len(self.y) / self.batch_size))

    def __getitem__(self, index):
        return self.__data_generation(index)

    def on_epoch_end(self):
        self.indexes = np.arange(len(self.y))
        if self.shuffle == True:
            np.random.shuffle(self.indexes)

    def __data_generation(self, index):
        Xb = np.empty((self.batch_size, *self.X[index].shape))
        yb = np.empty((self.batch_size, 1))

        # naively use the same sample over and over again
        for s in range(0, self.batch_size):
            Xb[s] = self.X[index]
            yb[s] = self.y[index]
        return Xb, yb


def task2_loader(hyperparameters):
    dataPath = '/Lab1/Lab5/HCP_lab/'
    train_subjects_list = ['599469', '599671', '601127'] # your choice of 3 training subjects
    val_subjects_list = ['613538']# your choice of 1 validation subjects
    bundles_list = ['CST_left', 'CST_right']
    x_train, y_train = load_streamlines(dataPath, train_subjects_list, bundles_list, hyperparameters['n_tracts_per_bundle'])
    x_val, y_val = load_streamlines(dataPath, val_subjects_list, bundles_list, hyperparameters['n_tracts_per_bundle'])
    train_data_generator = MyBatchGenerator(x_train, y_train, batch_size=1)
    test_data_generator = MyBatchGenerator(x_val, y_val, batch_size=1)
    return train_data_generator, test_data_generator


def load_streamlines(dataPath, subject_ids, bundles, n_tracts_per_bundle):
    X = []
    y = []
    for i in range(len(subject_ids)):
        for c in range((len(bundles))):
            filename = dataPath + subject_ids[i] + '/' + bundles[c] + '.trk'
            tfile = nib.streamlines.load(filename)
            streamlines = tfile.streamlines
            n_tracts_total = len(streamlines)
            ix_tracts = np.random.choice(range(n_tracts_total), n_tracts_per_bundle, replace=False)
            streamlines_data = streamlines.data
            streamlines_offsets = streamlines._offsets
            for j in range(n_tracts_per_bundle):
                ix_j = ix_tracts[j]
                offset_start = streamlines_offsets[ix_j]
                if ix_j < (n_tracts_total - 1):
                    offset_end = streamlines_offsets[ix_j + 1]
                    streamline_j = streamlines_data[offset_start:offset_end]
                else:
                    streamline_j = streamlines_data[offset_start:]
                X.append(np.asarray(streamline_j))
                y.append(c)
    return X, y


def get_data(task_num, hyperparameter):
    if task_num == "1a" or task_num == "1b":
        return task1_loader(hyperparameter)
    elif task_num == "2a" or task_num == "2b":
        return task2_loader(hyperparameter)
    elif task_num == "3a":
        return get_data_with_generator_on_the_fly(hyperparameter)