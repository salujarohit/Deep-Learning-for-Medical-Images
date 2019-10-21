import numpy as np
import os
from skimage.io import imread
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


class MyGenerator(Sequence):
    def __init__(self, image_names, mask_names, hyperparameters, step_num, fold_num, generator_type):
        self.hyperparameters = hyperparameters
        self.image_filenames = image_names
        self.mask_filenames = mask_names
        self.step_num = step_num
        self.fold_num = fold_num
        self.generator_type = generator_type
        self.indices = np.arange(len(self.mask_filenames))

    def __len__(self):
        return int(np.ceil(len(self.image_filenames) / float(self.hyperparameters['batch_size'])))

    @staticmethod
    def to_binary(img, threshold):
        img[img < threshold] = 0
        img[img >= threshold] = 1
        return img

    @staticmethod
    def to_categorical(img, num_classes):
        shape = list(img.shape)
        shape[2] = num_classes
        mask = np.zeros(shape)
        for i, value in enumerate(np.unique(img)):
            mask[(img == value)[:, :, 0], i] = 1
        return mask

    def normalize(self,image):
        pass

    @staticmethod
    def load_step_prediction(s_step, fold_num, fold_len, idx, batch_size, generator_type, data_shape):
        save_path = os.path.join(os.getcwd(), 'models')
        if os.path.isfile(os.path.join(save_path, 'posterior_unet_step' + str(s_step - 1) + '.npy')):
            predictions = np.load(os.path.join(save_path, 'posterior_unet_step' + str(s_step - 1) + '.npy'),
                                  allow_pickle=True)
            output_pred = np.zeros(data_shape)
            if generator_type == "testing":
                upper_limit = min((fold_num * fold_len) + ((idx + 1) * batch_size), ((fold_num + 1) * fold_len))
                output_indices = list(np.arange((fold_num * fold_len) + (idx * batch_size), upper_limit))
            else:
                test_indices = list(np.arange((fold_num * fold_len), ((fold_num + 1) * fold_len)))
                all_indices = list(np.arange(len(predictions)))
                train_indices = list(set(all_indices) - set(test_indices))
                output_indices = train_indices[idx * batch_size:(idx + 1) * batch_size]
            for i, ind in enumerate(output_indices):
                output_pred[i] = predictions[ind]
        else:
            output_pred = np.full(data_shape, .5)
        return output_pred

    def get_x(self, idx):
        indices = self.indices[idx * self.hyperparameters['batch_size']:(idx + 1) * self.hyperparameters['batch_size']]
        batch_x = self.image_filenames[indices]
        x = np.array([resize(imread(file), (self.hyperparameters['input_shape'][0], self.hyperparameters['input_shape'][0])) for file in batch_x])

        if self.hyperparameters['input_shape'][2] == 1:
            x = np.expand_dims(x, axis=3)

        if self.hyperparameters.get('autocontext_step'):
            self.hyperparameters['fold_len'] = len(self.image_filenames) // self.hyperparameters['folds']
            last_step_pred = self.load_step_prediction(self.step_num,
                                                       self.fold_num, self.hyperparameters['fold_len'],
                                                       idx, self.hyperparameters['batch_size'], self.generator_type, x.shape)
            x = np.concatenate((x, last_step_pred), axis=-1)

        return x

    def get_y(self, idx):
        indices = self.indices[idx * self.hyperparameters['batch_size']:(idx + 1) * self.hyperparameters['batch_size']]
        batch_y = self.mask_filenames[indices]
        y_list = []
        for file_name in batch_y:
            mask = imread(file_name, as_gray=False)
            mask = resize(mask, (self.hyperparameters['input_shape'][0], self.hyperparameters['input_shape'][1]), order=0, anti_aliasing=False,
                          preserve_range=True)
            if self.hyperparameters['input_shape'][2] == 1:
                mask = np.expand_dims(mask, axis=3)

            if self.hyperparameters['last_layer_units'] == 1:
                mask = self.to_binary(mask, self.hyperparameters['threshold'])
            else:
                mask = self.to_categorical(mask, num_classes=self.hyperparameters['last_layer_units'])
            y_list.append(mask)
        y = np.array(y_list)

        return y

    @staticmethod
    def combine_generators(gen1, gen2, gen3=None):
        batch_x = gen1.next()
        batch_y = gen2.next()
        if gen3:
            batch_w = gen3.next()
            return [batch_x, batch_w], batch_y
        else:
            return batch_x, batch_y

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

    def __getitem__(self, idx):
        x = self.get_x(idx)
        y = self.get_y(idx)
        image_data_generator = self.hyperparameters['generator'] if self.hyperparameters.get('generator') else {}
        if 'rescale' in image_data_generator:
            mask_generator_features = {i: image_data_generator[i] for i in image_data_generator if i != 'rescale'}
        else:
            mask_generator_features = image_data_generator
        x_data_gen = ImageDataGenerator(**image_data_generator)
        y_data_gen = ImageDataGenerator(**mask_generator_features)

        image_data_gen = x_data_gen.flow(x, batch_size=self.hyperparameters['batch_size'], seed=100)
        mask_data_gen = y_data_gen.flow(y, batch_size=self.hyperparameters['batch_size'], seed=100)

        mask_boundary_data_gen = None
        if self.hyperparameters.get('use_weight_maps'):
            mask_boundaries = self.get_mask_boundary(y)
            mask_boundary_data_gen = y_data_gen.flow(mask_boundaries, batch_size=self.hyperparameters['batch_size'], seed=100)

        return self.combine_generators(image_data_gen, mask_data_gen, mask_boundary_data_gen)


class DataLoader:
    def __init__(self, hyperparameters):
        self.hyperparameters = hyperparameters
        self.load_filenames()
        if self.hyperparameters.get('folds'):
            self.split_data_folds()
        else:
            self.split_data()

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

        training_batch_generator = MyGenerator(x_train, y_train, self.hyperparameters, step_num=step_num,
                                               fold_num=fold_num, generator_type="training")
        validation_batch_generator = MyGenerator(x_val, y_val, self.hyperparameters, step_num=step_num,
                                               fold_num=fold_num, generator_type="testing")

        return training_batch_generator, validation_batch_generator


class MyPredictionGenerator(Sequence):
    def __init__(self, hyperparameters, step_num=None, fold_num=None, generator_type="testing", len=90):
        self.hyperparameters = hyperparameters
        self.step_num = step_num
        self.fold_num = fold_num
        image_path = os.path.join(os.getcwd(), os.path.join(self.hyperparameters['data_path'], 'test_images'))
        self.image_filenames = np.array([os.path.join(image_path, file) for file in os.listdir(image_path)])
        self.image_filenames.sort()
        self.len = len

    def __len__(self):
        return int(self.len)

    def normalize(self,image):
        pass


    def get_case_indices(self, case_id):
        indices = []
        for i,image_name in enumerate(self.image_filenames):
            if case_id in image_name:
                indices.append(i)
        return indices
    # @staticmethod
    # def load_step_prediction(s_step, fold_num, fold_len, idx, batch_size, generator_type, data_shape):
    #     save_path = os.path.join(os.getcwd(), 'models')
    #     if os.path.isfile(os.path.join(save_path, 'posterior_unet_step' + str(s_step - 1) + '.npy')):
    #         predictions = np.load(os.path.join(save_path, 'posterior_unet_step' + str(s_step - 1) + '.npy'),
    #                               allow_pickle=True)
    #         output_pred = np.zeros(data_shape)
    #         if generator_type == "testing":
    #             upper_limit = min((fold_num * fold_len) + ((idx + 1) * batch_size), ((fold_num + 1) * fold_len))
    #             output_indices = list(np.arange((fold_num * fold_len) + (idx * batch_size), upper_limit))
    #         else:
    #             test_indices = list(np.arange((fold_num * fold_len), ((fold_num + 1) * fold_len)))
    #             all_indices = list(np.arange(len(predictions)))
    #             train_indices = list(set(all_indices) - set(test_indices))
    #             output_indices = train_indices[idx * batch_size:(idx + 1) * batch_size]
    #         for i, ind in enumerate(output_indices):
    #             output_pred[i] = predictions[ind]
    #     else:
    #         output_pred = np.full(data_shape, .5)
    #     return output_pred

    def __getitem__(self, idx):
        idx = str(idx).zfill(5)
        indices = self.get_case_indices(idx)
        if len(indices) == 0:
            return None, None
        batch_x = self.image_filenames[indices]
        original_shape = imread(batch_x[0]).shape
        x = np.array(
            [resize(imread(file), (self.hyperparameters['input_shape'][0], self.hyperparameters['input_shape'][0])) for
             file in batch_x])

        if self.hyperparameters['input_shape'][2] == 1:
            x = np.expand_dims(x, axis=3)

        # if self.hyperparameters.get('autocontext_step'):
        #     self.hyperparameters['fold_len'] = len(self.image_filenames) // self.hyperparameters['folds']
        #     last_step_pred = self.load_step_prediction(self.step_num,
        #                                                self.fold_num, self.hyperparameters['fold_len'],
        #                                                idx, self.hyperparameters['batch_size'], self.generator_type,
        #                                                x.shape)
        #     x = np.concatenate((x, last_step_pred), axis=-1)

        return x, original_shape




