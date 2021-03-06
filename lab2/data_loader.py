
import numpy as np
import os
from random import shuffle
from skimage.io import imread
from skimage.color import rgb2gray
from skimage.transform import resize
from tensorflow.python.keras.preprocessing.image import ImageDataGenerator, array_to_img

# Data Loader
# Assigning labels two images; those images contains pattern1 in their filenames
# will be labeled as class 0 and those with pattern2 will be labeled as class 1.
def gen_labels(im_name, pattern):
    for i,pat in enumerate(pattern):
        if pat in im_name:
            Label = np.array([i])
            break
    return Label


# reading and resizing the training images with their corresponding labels
def train_data(train_data_path, train_list, input_shape, pattern):
    train_img = []
    for i in range(len(train_list)):
        image_name = train_list[i]
        img = imread(os.path.join(train_data_path, image_name))
        if input_shape[2] == 1:
            img = rgb2gray(img)
            img = np.expand_dims(np.array(img), axis=3)
        img = resize(img, (input_shape[0], input_shape[1]), anti_aliasing=True).astype('float32')
        train_img.append([img, gen_labels(image_name, pattern)])

        if i % 200 == 0:
            print('Reading: {0}/{1}  of train images'.format(i, len(train_list)))

    shuffle(train_img)
    return train_img


# reading and resizing the testing images with their corresponding labels
def test_data(test_data_path, test_list, input_shape, pattern):
    test_img = []
    for i in range(len(test_list)):
        image_name = test_list[i]
        img = imread(os.path.join(test_data_path, image_name))
        if input_shape[2] == 1:
            img = rgb2gray(img)
            img = np.expand_dims(np.array(img), axis=3)
        img = resize(img, (input_shape[0], input_shape[1]), anti_aliasing=True).astype('float32')
        test_img.append([img, gen_labels(image_name, pattern)])

        if i % 100 == 0:
            print('Reading: {0}/{1} of test images'.format(i, len(test_list)))

    shuffle(test_img)
    return test_img


# Instantiating images and labels for the model.
def get_train_test_data(train_data_path, test_data_path, train_list, test_list, input_shape, pattern):
    Train_data = train_data(train_data_path, train_list, input_shape, pattern)
    Test_data = test_data(test_data_path, test_list, input_shape, pattern)

    Train_Img = np.zeros((len(train_list), input_shape[0], input_shape[1], input_shape[2]), dtype=np.float32)
    Test_Img = np.zeros((len(test_list), input_shape[0], input_shape[1], input_shape[2]), dtype=np.float32)

    Train_Label = np.zeros((len(train_list)), dtype=np.int32)
    Test_Label = np.zeros((len(test_list)), dtype=np.int32)

    for i in range(len(train_list)):
        Train_Img[i] = Train_data[i][0]
        Train_Label[i] = Train_data[i][1]


    for j in range(len(test_list)):
        Test_Img[j] = Test_data[j][0]
        Test_Label[j] = Test_data[j][1]


    return Train_Img, Test_Img, Train_Label, Test_Label


def get_data(hyperparameters):
    train_data_path = os.path.join(os.path.join(os.getcwd(), hyperparameters['data_path']), 'train')
    test_data_path = os.path.join(os.path.join(os.getcwd(), hyperparameters['data_path']), 'test')
    train_list = os.listdir(train_data_path)
    test_list = os.listdir(test_data_path)
    x_train, x_test, y_train, y_test = get_train_test_data(
        train_data_path, test_data_path,
        train_list, test_list, hyperparameters['input_shape'], hyperparameters['pattern'])
    print(x_train.shape, x_test.shape)
    return x_train, x_test, y_train, y_test


def get_number_images(data_path):
    number_images = 0
    target_list = os.listdir(data_path)
    for target in target_list:
        number_images += len(os.listdir(os.path.join(data_path, target)))

    return number_images


def get_data_with_generator(hyperparameters):
    train_data_path = os.path.join(os.path.join(os.getcwd(), hyperparameters['data_path']), 'train')
    test_data_path = os.path.join(os.path.join(os.getcwd(), hyperparameters['data_path']), 'validation')
    num_train = get_number_images(train_data_path)
    num_test = get_number_images(test_data_path)
    image_gen_train = ImageDataGenerator(
        rescale=hyperparameters['gen']['rescale'],
        rotation_range=hyperparameters['gen']['rotation_range'],
        width_shift_range=hyperparameters['gen']['width_shift_range'],
        height_shift_range=hyperparameters['gen']['height_shift_range'],
        horizontal_flip=hyperparameters['gen']['horizontal_flip'],
        zoom_range=hyperparameters['gen']['zoom_range'],
    )
    image_gen_test = ImageDataGenerator(
        rescale=hyperparameters['gen_test']['rescale'],
        rotation_range=hyperparameters['gen_test']['rotation_range'],
        width_shift_range=hyperparameters['gen_test']['width_shift_range'],
        height_shift_range=hyperparameters['gen_test']['height_shift_range'],
        horizontal_flip=hyperparameters['gen_test']['horizontal_flip'],
        zoom_range=hyperparameters['gen_test']['zoom_range'],
    )
    if hyperparameters['dense_units'][-1] == 1:
        class_mode = "binary"
        color_mode = "grayscale"
    else:
        class_mode = "categorical"
        color_mode = "rgb"
    if hyperparameters['input_shape'][2] == 1:
        color_mode = "grayscale"
    else:
        color_mode = "rgb"

    train_data_gen = image_gen_train.flow_from_directory(batch_size=hyperparameters['batch_size'],
                                                   directory=train_data_path,
                                                   shuffle=True, class_mode=class_mode,
                                                   target_size=(hyperparameters['input_shape'][0], hyperparameters['input_shape'][1]), color_mode = color_mode)

    test_data_gen = image_gen_test.flow_from_directory(batch_size=hyperparameters['batch_size'],
                                                         directory=test_data_path,
                                                         shuffle=True, class_mode=class_mode,
                                                         target_size=(hyperparameters['input_shape'][0],
                                                                      hyperparameters['input_shape'][1]), color_mode = color_mode)


    return train_data_gen, test_data_gen, num_train, num_test


def load_data(hyperparameters):
    if hyperparameters['use_gen']:
        return get_data_with_generator(hyperparameters)
    else:
        return get_data(hyperparameters)

