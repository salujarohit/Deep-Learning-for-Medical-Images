import numpy as np
import matplotlib.pyplot as plt
import os
from skimage.io import imread
from skimage.transform import resize
import cv2

try:
    import tensorflow as tf
    from tensorflow.keras.models import Sequential, load_model, Model
    from tensorflow.keras.layers import Input, Add, Flatten, Conv2D, MaxPooling2D, Activation, Dropout, Dense, BatchNormalization, \
        SpatialDropout2D, GlobalAveragePooling2D, ZeroPadding2D, AveragePooling2D
    from tensorflow.keras.optimizers import Adam, SGD, RMSprop
    from tensorflow.keras.applications import VGG16, InceptionV3
    from tensorflow.keras import backend as K
    from tensorflow.keras.initializers import glorot_uniform



except:
    import tensorflow as tf
    from tensorflow.python.keras.models import Sequential, load_model, Model
    from tensorflow.python.keras.layers.core import Input, Add, Flatten, Conv2D, MaxPooling2D, Activation, Dropout, Dense, \
        BatchNormalization, SpatialDropout2D, GlobalAveragePooling2D, ZeroPadding2D, AveragePooling2D
    from tensorflow.python.keras.optimizers import Adam, SGD, RMSprop
    from tensorflow.python.keras.applications import VGG16, InceptionV3
    from tensorflow.python.keras import backend as K
    from tensorflow.python.keras.initializers import glorot_uniform


# AlexNet Model
def model_AlexNet(hyperparameters):
    model = Sequential()

    model.add(Conv2D(filters=hyperparameters['base'], input_shape=hyperparameters['input_shape'],
                     kernel_size=(3, 3), strides=(1, 1), padding='same'))
    if hyperparameters['batch_norm']:
        model.add(BatchNormalization())
    model.add(Activation('relu'))
    if len(hyperparameters['spatial_dropout']) > 0:
        model.add(SpatialDropout2D(hyperparameters['spatial_dropout'][0]))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Conv2D(filters=hyperparameters['base'] * 2, kernel_size=(3, 3), strides=(1, 1), padding='same'))
    if hyperparameters['batch_norm']:
        model.add(BatchNormalization())
    model.add(Activation('relu'))
    if len(hyperparameters['spatial_dropout']) > 1:
        model.add(SpatialDropout2D(hyperparameters['spatial_dropout'][1]))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Conv2D(filters=hyperparameters['base'] * 4, kernel_size=(3, 3), strides=(1, 1), padding='same'))
    if hyperparameters['batch_norm']:
        model.add(BatchNormalization())
    model.add(Activation('relu'))
    model.add(Conv2D(filters=hyperparameters['base'] * 4, kernel_size=(3, 3), strides=(1, 1), padding='same'))
    if hyperparameters['batch_norm']:
        model.add(BatchNormalization())
    model.add(Activation('relu'))
    model.add(Conv2D(filters=hyperparameters['base'] * 2, kernel_size=(3, 3), strides=(1, 1), padding='same'))
    if hyperparameters['batch_norm']:
        model.add(BatchNormalization())
    model.add(Activation('relu'))
    if len(hyperparameters['spatial_dropout']) > 2:
        model.add(SpatialDropout2D(hyperparameters['spatial_dropout'][2]))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Flatten())
    for i, unit in enumerate(hyperparameters['dense_units']):
        model.add(Dense(unit))
        model.add(Activation(hyperparameters['dense_activation'][i]))
        if len(hyperparameters['dropout']) > i:
            model.add(Dropout(hyperparameters['dropout'][i]))

    print(model.summary())
    model.compile(loss=hyperparameters['loss'],
                  optimizer=hyperparameters['optimizer'](lr=hyperparameters['lr']),
                  metrics=hyperparameters['metrics'])

    return model


def model_vgg16(hyperparameters):
    model = Sequential()
    model.add(Conv2D(hyperparameters['base'], (3, 3), input_shape=hyperparameters['input_shape'], padding='same'))
    if hyperparameters['batch_norm']:
        model.add(BatchNormalization())
    model.add(Activation('relu'))
    model.add(Conv2D(hyperparameters['base'], (3, 3), padding='same'))
    if hyperparameters['batch_norm']:
        model.add(BatchNormalization())
    model.add(Activation('relu'))
    if len(hyperparameters['spatial_dropout']) > 0:
        model.add(SpatialDropout2D(hyperparameters['spatial_dropout'][0]))
    model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2))),
    model.add(Conv2D(hyperparameters['base'] * 2, (3, 3), padding='same'))
    if hyperparameters['batch_norm']:
        model.add(BatchNormalization())
    model.add(Activation('relu'))
    model.add(Conv2D(hyperparameters['base'] * 2, (3, 3), padding='same'))
    if hyperparameters['batch_norm']:
        model.add(BatchNormalization())
    model.add(Activation('relu'))
    if len(hyperparameters['spatial_dropout']) > 1:
        model.add(SpatialDropout2D(hyperparameters['spatial_dropout'][1]))
    model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))
    model.add(Conv2D(hyperparameters['base'] * 4, (3, 3), padding='same'))
    if hyperparameters['batch_norm']:
        model.add(BatchNormalization())
    model.add(Activation('relu'))
    model.add(Conv2D(hyperparameters['base'] * 4, (3, 3), padding='same'))
    if hyperparameters['batch_norm']:
        model.add(BatchNormalization())
    model.add(Activation('relu'))
    model.add(Conv2D(hyperparameters['base'] * 4, (3, 3), padding='same'))
    if hyperparameters['batch_norm']:
        model.add(BatchNormalization())
    model.add(Activation('relu'))
    if len(hyperparameters['spatial_dropout']) > 2:
        model.add(SpatialDropout2D(hyperparameters['spatial_dropout'][2]))
    model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))
    model.add(Conv2D(hyperparameters['base'] * 8, (3, 3), padding='same'))
    if hyperparameters['batch_norm']:
        model.add(BatchNormalization())
    model.add(Activation('relu'))
    model.add(Conv2D(hyperparameters['base'] * 8, (3, 3), padding='same'))
    if hyperparameters['batch_norm']:
        model.add(BatchNormalization())
    model.add(Activation('relu'))
    model.add(Conv2D(hyperparameters['base'] * 8, (3, 3), padding='same'))
    if hyperparameters['batch_norm']:
        model.add(BatchNormalization())
    model.add(Activation('relu'))
    if len(hyperparameters['spatial_dropout']) > 3:
        model.add(SpatialDropout2D(hyperparameters['spatial_dropout'][3]))
    MaxPooling2D(pool_size=(2, 2), strides=(2, 2)),
    model.add(Conv2D(hyperparameters['base'] * 8, (3, 3), padding='same'))
    if hyperparameters['batch_norm']:
        model.add(BatchNormalization())
    model.add(Activation('relu'))
    model.add(Conv2D(hyperparameters['base'] * 8, (3, 3), padding='same'))
    if hyperparameters['batch_norm']:
        model.add(BatchNormalization())
    model.add(Activation('relu'))
    model.add(Conv2D(hyperparameters['base'] * 8, (3, 3), padding='same'))
    if hyperparameters['batch_norm']:
        model.add(BatchNormalization())
    model.add(Activation('relu'))
    if len(hyperparameters['spatial_dropout']) > 4:
        model.add(SpatialDropout2D(hyperparameters['spatial_dropout'][4]))
    model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))
    model.add(Flatten())
    for i, unit in enumerate(hyperparameters['dense_units']):
        model.add(Dense(unit))
        model.add(Activation(hyperparameters['dense_activation'][i]))
        if len(hyperparameters['dropout']) > i:
            model.add(Dropout(hyperparameters['dropout'][i]))

    print(model.summary())
    model.compile(loss=hyperparameters['loss'],
                  optimizer=hyperparameters['optimizer'](lr=hyperparameters['lr']),
                  metrics=hyperparameters['metrics'])

    return model


def identity_block(X, f, filters, stage, block):
    # Defining name basis
    conv_name_base = 'res' + str(stage) + block + '_branch'
    bn_name_base = 'bn' + str(stage) + block + '_branch'

    # Retrieve Filters
    F1, F2, F3 = filters

    # Save the input value
    X_shortcut = X

    # First component of main path
    X = Conv2D(filters=F1, kernel_size=(1, 1), strides=(1, 1), padding='valid', name=conv_name_base + '2a',
               kernel_initializer=glorot_uniform(seed=0))(X)
    X = BatchNormalization(axis=3, name=bn_name_base + '2a')(X)
    X = Activation('relu')(X)

    # Second component of main path
    X = Conv2D(filters=F2, kernel_size=(f, f), strides=(1, 1), padding='same', name=conv_name_base + '2b',
               kernel_initializer=glorot_uniform(seed=0))(X)
    X = BatchNormalization(axis=3, name=bn_name_base + '2b')(X)
    X = Activation('relu')(X)

    # Third component of main path
    X = Conv2D(filters=F3, kernel_size=(1, 1), strides=(1, 1), padding='valid', name=conv_name_base + '2c',
               kernel_initializer=glorot_uniform(seed=0))(X)
    X = BatchNormalization(axis=3, name=bn_name_base + '2c')(X)

    # Final step: Add shortcut value to main path, and pass it through a RELU activation
    X = Add()([X, X_shortcut])
    X = Activation('relu')(X)

    return X

def convolutional_block(X, f, filters, stage, block, s=2):

    # Defining name basis
    conv_name_base = 'res' + str(stage) + block + '_branch'
    bn_name_base = 'bn' + str(stage) + block + '_branch'

    # Retrieve Filters
    F1, F2, F3 = filters

    # Save the input value
    X_shortcut = X

    ##### MAIN PATH #####
    # First component of main path
    X = Conv2D(filters=F1, kernel_size=(1, 1), strides=(s, s), padding='valid', name=conv_name_base + '2a', kernel_initializer=glorot_uniform(seed=0))(X)
    X = BatchNormalization(axis=3, name=bn_name_base + '2a')(X)
    X = Activation('relu')(X)

    # Second component of main path
    X = Conv2D(filters=F2, kernel_size=(f, f), strides=(1, 1), padding='same', name=conv_name_base + '2b', kernel_initializer=glorot_uniform(seed=0))(X)
    X = BatchNormalization(axis=3, name=bn_name_base + '2b')(X)
    X = Activation('relu')(X)

    # Third component of main path
    X = Conv2D(filters=F3, kernel_size=(1, 1), strides=(1, 1), padding='valid', name=conv_name_base + '2c', kernel_initializer=glorot_uniform(seed=0))(X)
    X = BatchNormalization(axis=3, name=bn_name_base + '2c')(X)

    ##### SHORTCUT PATH ####
    X_shortcut = Conv2D(filters=F3, kernel_size=(1, 1), strides=(s, s), padding='valid', name=conv_name_base + '1', kernel_initializer=glorot_uniform(seed=0))(X_shortcut)
    X_shortcut = BatchNormalization(axis=3, name=bn_name_base + '1')(X_shortcut)

    # Final step: Add shortcut value to main path, and pass it through a RELU activation
    X = Add()([X, X_shortcut])
    X = Activation('relu')(X)

    return X


def model_ResNet50(hyperparameters):
    # Define the input as a tensor with shape input_shape
    X_input = Input(hyperparameters['input_shape'])

    # Zero-Padding
    X = ZeroPadding2D((3, 3))(X_input)

    # Stage 1
    X = Conv2D(64, (7, 7), strides=(2, 2), name='conv1', kernel_initializer=glorot_uniform(seed=0))(X)
    X = BatchNormalization(axis=3, name='bn_conv1')(X)
    X = Activation('relu')(X)
    X = MaxPooling2D((3, 3), strides=(2, 2))(X)

    # Stage 2
    X = convolutional_block(X, f=3, filters=[64, 64, 256], stage=2, block='a', s=1)
    X = identity_block(X, 3, [64, 64, 256], stage=2, block='b')
    X = identity_block(X, 3, [64, 64, 256], stage=2, block='c')

    # Stage 3
    X = convolutional_block(X, f=3, filters=[128, 128, 512], stage=3, block='a', s=2)
    X = identity_block(X, 3, [128, 128, 512], stage=3, block='b')
    X = identity_block(X, 3, [128, 128, 512], stage=3, block='c')
    X = identity_block(X, 3, [128, 128, 512], stage=3, block='d')

    # Stage 4
    X = convolutional_block(X, f=3, filters=[256, 256, 1024], stage=4, block='a', s=2)
    X = identity_block(X, 3, [256, 256, 1024], stage=4, block='b')
    X = identity_block(X, 3, [256, 256, 1024], stage=4, block='c')
    X = identity_block(X, 3, [256, 256, 1024], stage=4, block='d')
    X = identity_block(X, 3, [256, 256, 1024], stage=4, block='e')
    X = identity_block(X, 3, [256, 256, 1024], stage=4, block='f')

    # Stage 5
    X = convolutional_block(X, f=3, filters=[512, 512, 2048], stage=5, block='a', s=2)
    X = identity_block(X, 3, [512, 512, 2048], stage=5, block='b')
    X = identity_block(X, 3, [512, 512, 2048], stage=5, block='c')

    # AVGPOOL
    X = AveragePooling2D(pool_size=(2, 2), padding='same')(X)

    # Output layer
    X = Flatten()(X)
    for i, unit in enumerate(hyperparameters['dense_units']):
        X = Dense(unit, name='fc' + str(i), kernel_initializer=glorot_uniform(seed=0))(X)
        X = Activation(hyperparameters['dense_activation'][i])(X)
        if len(hyperparameters['dropout']) > i:
            X = Dropout(hyperparameters['dropout'][i])(X)

    # Create model
    model = Model(inputs=X_input, outputs=X, name='ResNet50')

    print(model.summary())
    model.compile(loss=hyperparameters['loss'],
                  optimizer=hyperparameters['optimizer'](lr=hyperparameters['lr']),
                  metrics=hyperparameters['metrics'])
    return model

def process_pretrained_layer(hyperparameters, i):
    if hyperparameters['pretrained_prop']['added_layers'][i] == "Dense":
        layer = Dense(int(hyperparameters['pretrained_prop']['added_layers_parameters'][i]))
    elif hyperparameters['pretrained_prop']['added_layers'][i] == "Flatten":
        layer = Flatten()
    elif hyperparameters['pretrained_prop']['added_layers'][i] == "GlobalAveragePooling2D":
        layer = GlobalAveragePooling2D()
    elif hyperparameters['pretrained_prop']['added_layers'][i] == "Activation":
        layer = Activation(hyperparameters['pretrained_prop']['added_layers_parameters'][i])
    elif hyperparameters['pretrained_prop']['added_layers'][i] == "Dropout":
        layer = Dropout(float(hyperparameters['pretrained_prop']['added_layers_parameters'][i]))
    return layer


def get_pretrained_model(hyperparameters):
    base = VGG16
    if hyperparameters['pretrained_prop']['pretrained_model'] == "inception":
        base = InceptionV3
    base_model = base(input_shape=hyperparameters['input_shape'],
                      include_top=hyperparameters['pretrained_prop']['include_top'],
                      weights=hyperparameters['pretrained_prop']['weights'])
    base_model.trainable = hyperparameters['pretrained_prop']['base_model_trainable']
    model = Sequential()
    model.add(base_model)
    for i, _ in enumerate(hyperparameters['pretrained_prop']['added_layers']):
        model.add(process_pretrained_layer(hyperparameters, i))

    print(model.summary())
    model.compile(loss=hyperparameters['loss'],
                  optimizer=hyperparameters['optimizer'](lr=hyperparameters['lr']),
                  metrics=hyperparameters['metrics'])
    return model


def get_model(hyperparameters):
    if hyperparameters['use_pretrained']:
        return get_pretrained_model(hyperparameters)
    elif hyperparameters['model'] == "alexnet":
        return model_AlexNet(hyperparameters)
    elif hyperparameters['model'] == "vgg16":
        return model_vgg16(hyperparameters)
    elif hyperparameters['model'] == "resnet":
        return model_ResNet50(hyperparameters)


def plot_history(History, task_number):

    if not os.path.isdir(os.path.join(os.getcwd(), 'results')):
        os.mkdir(os.path.join(os.getcwd(), 'results'))
    fig = plt.figure(figsize=(4, 4))
    plt.title("Learning curve")
    plt.plot(History.history["loss"], label="loss")
    plt.plot(History.history["val_loss"], label="val_loss")
    plt.plot(np.argmin(History.history["val_loss"]),
             np.min(History.history["val_loss"]),
             marker="x", color="r", label="best model")

    plt.xlabel("Epochs")
    plt.ylabel("Loss Value")
    plt.legend()
    result_path = os.path.join(os.path.join(os.getcwd(), 'results'), str(task_number) + '_loss.png')
    fig.savefig(result_path, dpi=fig.dpi)

    accuracy = ''
    val_accuracy = ''
    for key in History.history:
        if "val" not in key and "accuracy" in key:
            accuracy = key
        if "val" in key and "accuracy" in key:
            val_accuracy = key
    if accuracy != '' and val_accuracy != '':
        fig = plt.figure(figsize=(4, 4))
        plt.title("Accuracy curve")
        plt.plot(History.history[accuracy], label="accuracy")
        plt.plot(History.history[val_accuracy], label="val_accuracy")
        plt.plot(np.argmax(History.history[val_accuracy]),
                 np.max(History.history[val_accuracy]),
                 marker="x", color="r", label="best model")

        plt.xlabel("Epochs")
        plt.ylabel("Accuracy Value")
        plt.legend()
        result_path = os.path.join(os.path.join(os.getcwd(), 'results'), str(task_number) + '_accuracy.png')
        fig.savefig(result_path, dpi=fig.dpi)


def show_visualization(model, hyperparameters):
    Sample = '/Lab1/Lab2/Bone/train/AFF/14.jpg'
    Img = imread(Sample)
    Img = Img[:, :, 0]
    Img = Img / 255
    img_height, img_width = hyperparameters['input_shape'][0], hyperparameters['input_shape'][1]
    Img = resize(Img, (img_height, img_width), anti_aliasing=True).astype('float32')
    Img = np.expand_dims(Img, axis=2)
    Img = np.expand_dims(Img, axis=0)
    preds = model.predict(Img)
    class_idx = np.argmax(preds[0])
    print(class_idx)
    class_output = model.output[:, class_idx]
    last_conv_layer = model.get_layer("conv2d_12")
    print(last_conv_layer.output)
    grads = K.gradients(class_output, last_conv_layer.output)[0]
    pooled_grads = K.mean(grads, axis=(0, 1, 2))
    iterate = K.function([model.input], [pooled_grads, last_conv_layer.output[0]])
    pooled_grads_value, conv_layer_output_value = iterate([Img])
    for i in range(hyperparameters['base'] * 8):
        conv_layer_output_value[:, :, i] *= pooled_grads_value[i]

    heatmap = np.mean(conv_layer_output_value, axis=-1)
    heatmap = np.maximum(heatmap, 0)
    heatmap /= np.max(heatmap)
    # For visualization
    img = cv2.imread(Sample)
    img = cv2.resize(img, (512, 512), interpolation=cv2.INTER_AREA)
    heatmap = cv2.resize(heatmap, (img.shape[1], img.shape[0]))
    heatmap = np.uint8(255 * heatmap)
    heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)
    superimposed_img = cv2.addWeighted(img, 0.6, heatmap, 0.4, 0)
    fig = plt.figure()
    plt.imshow(img)
    result_path = os.path.join(os.path.join(os.getcwd(), 'results'), "Task 10_img.png")
    fig.savefig(result_path, dpi=fig.dpi)
    fig = plt.figure()
    plt.imshow(superimposed_img)
    result_path = os.path.join(os.path.join(os.getcwd(), 'results'), "Task 10_superimposed_img.png")
    fig.savefig(result_path, dpi=fig.dpi)

