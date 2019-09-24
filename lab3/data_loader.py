import numpy as np
import os
from random import shuffle
from skimage.io import imread
from skimage.color import rgb2gray
from skimage.transform import resize
import random
from  sklearn.model_selection import train_test_split
try:
    from tensorflow.keras.processing.image import ImageDataGenerator, array_to_img
    from tensorflow.keras.utils import Sequence
except:
    from tensorflow.python.keras.preprocessing.image import ImageDataGenerator, array_to_img
    from tensorflow.python.keras.utils import Sequence


def binarize(img, values=[255]):
    mask = np.zeros((img.shape))
    for value in values:
        mask[img == value] = value
    return mask

def get_data(hyperparameters):
    data_path = os.path.join(os.getcwd(), hyperparameters['data_path'])
    image_names = os.listdir(os.path.join(data_path, 'Image'))
    mask_names = os.listdir(os.path.join(data_path, 'Mask'))

    data = []
    for i, (image_name, mask_name) in enumerate(zip(image_names, mask_names)):
        img = imread(os.path.join(os.path.join(data_path, 'Image'), image_name))
        mask = imread(os.path.join(os.path.join(data_path, 'Mask'), mask_name))
        if hyperparameters['input_shape'][2] == 1:
            img = rgb2gray(img)
            img = np.expand_dims(np.array(img), axis=3)
            mask = rgb2gray(mask)
            mask = np.expand_dims(np.array(mask), axis=3)

        img = resize(img, (hyperparameters['input_shape'][0], hyperparameters['input_shape'][1]),
                     anti_aliasing=True).astype('float32')
        mask = resize(mask, (hyperparameters['input_shape'][0], hyperparameters['input_shape'][0]), order=0,
                     anti_aliasing=False, preserve_range=True)

        mask = binarize(mask, hyperparameters['binarize_values'])
        # mask[mask < 200] = 0
        # mask[mask >= 200] = 255
        mask /= 255

        data.append([img, mask])

        if i % 200 == 0:
            print('Reading: {0}/{1}  of train images'.format(len(data), len(image_names)))

    shuffle(data)

    x = np.zeros((len(data), hyperparameters['input_shape'][0], hyperparameters['input_shape'][1], hyperparameters['input_shape'][2]), dtype=np.float32)
    y = np.zeros((len(data), hyperparameters['input_shape'][0], hyperparameters['input_shape'][1], hyperparameters['input_shape'][2]), dtype=np.float32)

    for i in range(len(data)):
        x[i] = data[i][0]
        y[i] = data[i][1]

    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=hyperparameters['test_size'], random_state=1)

    return x_train, x_test, y_train, y_test

def combine_generators(gen1, gen2):
    while True:
        batch_x = next(gen1)
        batch_y = next(gen2)
        yield batch_x, batch_y

def get_data_with_generator(hyperparameters):
    x_train, x_test, y_train, y_test = get_data(hyperparameters)
    num_test = len(x_test)
    num_train = len(x_train)
    default_generator_parameters = {'rescale': 0, 'rotation_range': 0, 'width_shift_range': 0,
                      'height_shift_range': 0, 'horizontal_flip': False, 'zoom_range': 0}
    train_generator_parameters = default_generator_parameters if 'generator' not in hyperparameters else hyperparameters['generator']
    test_generator_parameters = default_generator_parameters if 'test_generator' not in hyperparameters else hyperparameters['test_generator']
    data_generator_train = ImageDataGenerator(**train_generator_parameters)
    data_generator_test = ImageDataGenerator(**test_generator_parameters)
    train_data_generator_image = data_generator_train.flow(x_train, batch_size=hyperparameters['batch_size'],
                                                           shuffle=True, seed=100)
    train_data_generator_mask = data_generator_train.flow(y_train, batch_size=hyperparameters['batch_size'],
                                                          shuffle=True, seed=100)
    test_data_generator_image = data_generator_test.flow(x_test, batch_size=hyperparameters['batch_size'],
                                                               shuffle=True, seed=100)
    test_data_generator_mask = data_generator_test.flow(y_test, batch_size=hyperparameters['batch_size'],
                                                              shuffle=True, seed=100)

    train_data_generator = combine_generators(train_data_generator_image, train_data_generator_mask)
    test_data_generator = combine_generators(test_data_generator_image, test_data_generator_mask)

    return train_data_generator, test_data_generator, num_train, num_test


def load_data(hyperparameters):
    # if 'generator' in hyperparameters:
    return get_data_with_generator(hyperparameters)
    # else:
    #     return get_data(hyperparameters)

class MyGenerator(Sequence):

    def __init__(self, image_filenames, mask_filenames, batch_size, input_shape, generator_features):
        self.image_filenames, self.mask_filenames = image_filenames, mask_filenames
        self.batch_size = batch_size
        self.input_shape = input_shape
        self.generator_features = generator_features

    def __len__(self):
        return int(np.ceil(len(self.image_filenames) / float(self.batch_size)))

    def __getitem__(self, idx):
        batch_x = self.image_filenames[idx * self.batch_size:(idx + 1) * self.batch_size]
        batch_y = self.mask_filenames[idx * self.batch_size:(idx + 1) * self.batch_size]
        image_gen = ImageDataGenerator(** self.generator_features)
        if self.input_shape[2] == 1:
            as_gray = True
        else:
            as_gray = False
        x = np.array(
            [resize(imread(file_name, as_gray=as_gray), (self.input_shape[0], self.input_shape[1])) for file_name in
             batch_x])
        if self.input_shape[2] == 1:
            x = np.expand_dims(x, axis=3)
        y = np.array(
            [resize(imread(file_name, as_gray=as_gray), (self.input_shape[0], self.input_shape[1])) for file_name in
             batch_y])
        if self.input_shape[2] == 1:
            y = np.expand_dims(y, axis=3)

        data_gen = image_gen.flow(x, y, batch_size=self.batch_size)
        for batch_x, batch_y in data_gen:
            return batch_x, batch_y


def get_data_with_generator_on_the_fly(hyperparameters):
    image_path = os.path.join(os.getcwd(), os.path.join(hyperparameters['data_path'], 'Image'))
    mask_path = os.path.join(os.getcwd(), os.path.join(hyperparameters['data_path'], 'Mask'))
    image_names = [os.path.join(image_path, file) for file in os.listdir(image_path)]
    mask_names = [os.path.join(mask_path, file) for file in os.listdir(mask_path)]
    # paired = list(zip(image_names, mask_names))
    # random.shuffle(paired)
    # separated = [list(t) for t in zip(*paired)]
    # shuffled_x = separated[0]
    # shuffled_y = separated[1]
    # x_train, x_test, y_train, y_test = train_test_split(shuffled_x, shuffled_y, test_size=hyperparameters['test_size'],
    #                                                     random_state=1)
    x_train, x_test, y_train, y_test = train_test_split(image_names, mask_names, test_size=hyperparameters['test_size'],
                                                        random_state=1)
    data_generator = hyperparameters['generator'] if 'generator' in hyperparameters else {}
    training_batch_generator = MyGenerator(x_train, y_train, hyperparameters['batch_size'],
                                           hyperparameters['input_shape'], data_generator)
    validation_batch_generator = MyGenerator(x_test, y_test, hyperparameters['batch_size'],
                                             hyperparameters['input_shape'], data_generator)
    return training_batch_generator, validation_batch_generator, len(y_train), len(y_test)
