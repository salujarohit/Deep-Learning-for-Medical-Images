import numpy as np
import matplotlib.pyplot as plt
import os

try:
    import tensorflow as tf
    from tensorflow.keras.models import Sequential, load_model, Model
    from tensorflow.keras.layers import Input, Add, Flatten, Conv2D, MaxPooling2D, Activation, Dropout, Dense,\
        BatchNormalization, SpatialDropout2D, GlobalAveragePooling2D, ZeroPadding2D, AveragePooling2D,\
        Conv2DTranspose, concatenate
    from tensorflow.keras.optimizers import Adam, SGD, RMSprop
    from tensorflow.keras.applications import VGG16, InceptionV3
    from tensorflow.keras import backend as K
    from tensorflow.keras.initializers import glorot_uniform
except:
    import tensorflow as tf
    from tensorflow.python.keras.models import Sequential, load_model, Model
    from tensorflow.python.keras.layers.core import Input, Add, Flatten, Conv2D, MaxPooling2D, Activation, Dropout,\
        Dense, BatchNormalization, SpatialDropout2D, GlobalAveragePooling2D, ZeroPadding2D, AveragePooling2D,\
        Conv2DTranspose, concatenate
    from tensorflow.python.keras.optimizers import Adam, SGD, RMSprop
    from tensorflow.python.keras.applications import VGG16, InceptionV3
    from tensorflow.python.keras import backend as K
    from tensorflow.python.keras.initializers import glorot_uniform


def get_unet(hyperparameters):
    inputs = Input(hyperparameters['input_shape'])
    conv1 = Conv2D(hyperparameters['base'], (3, 3), padding='same')(inputs)
    if hyperparameters['batch_norm']:
        conv1 = BatchNormalization()(conv1)
    conv1 = Activation('relu')(conv1)
    conv1 = Conv2D(hyperparameters['base'], (3, 3), padding='same')(conv1)
    if hyperparameters['batch_norm']:
        conv1 = BatchNormalization()(conv1)
    conv1 = Activation('relu')(conv1)
    if hyperparameters['dropout'] != 0:
        conv1 = Dropout(hyperparameters['dropout'])(conv1)
    pool1 = MaxPooling2D(pool_size=(2, 2))(conv1)
    conv2 = Conv2D(hyperparameters['base'] * 2, (3, 3), padding='same')(pool1)
    if hyperparameters['batch_norm']:
        conv2 = BatchNormalization()(conv2)
    conv2 = Activation('relu')(conv2)
    conv2 = Conv2D(hyperparameters['base'] * 2, (3, 3), padding='same')(conv2)
    if hyperparameters['batch_norm']:
        conv2 = BatchNormalization()(conv2)
    conv2 = Activation('relu')(conv2)
    if hyperparameters['dropout'] != 0:
        conv2 = Dropout(hyperparameters['dropout'])(conv2)
    pool2 = MaxPooling2D(pool_size=(2, 2))(conv2)
    conv3 = Conv2D(hyperparameters['base'] * 4, (3, 3), padding='same')(pool2)
    if hyperparameters['batch_norm']:
        conv3 = BatchNormalization()(conv3)
    conv3 = Activation('relu')(conv3)
    conv3 = Conv2D(hyperparameters['base'] * 4, (3, 3), padding='same')(conv3)
    if hyperparameters['batch_norm']:
        conv3 = BatchNormalization()(conv3)
    conv3 = Activation('relu')(conv3)
    if hyperparameters['dropout'] != 0:
        conv3 = Dropout(hyperparameters['dropout'])(conv3)
    pool3 = MaxPooling2D(pool_size=(2, 2))(conv3)
    conv4 = Conv2D(hyperparameters['base'] * 8, (3, 3), padding='same')(pool3)
    if hyperparameters['batch_norm']:
        conv4 = BatchNormalization()(conv4)
    conv4 = Activation('relu')(conv4)
    conv4 = Conv2D(hyperparameters['base'] * 8, (3, 3), padding='same')(conv4)
    if hyperparameters['batch_norm']:
        conv4 = BatchNormalization()(conv4)
    conv4 = Activation('relu')(conv4)
    if hyperparameters['dropout'] != 0:
        conv4 = Dropout(hyperparameters['dropout'])(conv4)
    pool4 = MaxPooling2D(pool_size=(2, 2))(conv4)
    conv5 = Conv2D(hyperparameters['base'] * 16, (3, 3), padding='same')(pool4)
    if hyperparameters['batch_norm']:
        conv5 = BatchNormalization()(conv5)
    conv5 = Activation('relu')(conv5)
    conv5 = Conv2D(hyperparameters['base'] * 16, (3, 3), padding='same')(conv5)
    if hyperparameters['batch_norm']:
        conv5 = BatchNormalization()(conv5)
    conv5 = Activation('relu')(conv5)
    up6 = concatenate([Conv2DTranspose(hyperparameters['base'] * 8, (2, 2), strides=(2, 2), padding='same')(conv5), conv4], axis=3)
    conv6 = Conv2D(hyperparameters['base'] * 8, (3, 3), padding='same')(up6)
    if hyperparameters['batch_norm']:
        conv6 = BatchNormalization()(conv6)
    conv6 = Activation('relu')(conv6)
    conv6 = Conv2D(hyperparameters['base'] * 8, (3, 3), padding='same')(conv6)
    if hyperparameters['batch_norm']:
        conv6 = BatchNormalization()(conv6)
    conv6 = Activation('relu')(conv6)
    up7 = concatenate([Conv2DTranspose(128, (2, 2), strides=(2, 2), padding='same')(conv6), conv3], axis=3)
    conv7 = Conv2D(hyperparameters['base'] * 4, (3, 3), padding='same')(up7)
    if hyperparameters['batch_norm']:
        conv7 = BatchNormalization()(conv7)
    conv7 = Activation('relu')(conv7)
    conv7 = Conv2D(hyperparameters['base'] * 4, (3, 3), padding='same')(conv7)
    if hyperparameters['batch_norm']:
        conv7 = BatchNormalization()(conv7)
    conv7 = Activation('relu')(conv7)
    up8 = concatenate([Conv2DTranspose(hyperparameters['base'] * 2, (2, 2), strides=(2, 2), padding='same')(conv7), conv2], axis=3)
    conv8 = Conv2D(hyperparameters['base'] * 2, (3, 3), padding='same')(up8)
    if hyperparameters['batch_norm']:
        conv8 = BatchNormalization()(conv8)
    conv8 = Activation('relu')(conv8)
    conv8 = Conv2D(hyperparameters['base'] * 2, (3, 3), padding='same')(conv8)
    if hyperparameters['batch_norm']:
        conv8 = BatchNormalization()(conv8)
    conv8 = Activation('relu')(conv8)
    up9 = concatenate([Conv2DTranspose(hyperparameters['base'], (2, 2), strides=(2, 2), padding='same')(conv8), conv1], axis=3)
    conv9 = Conv2D(hyperparameters['base'], (3, 3), padding='same')(up9)
    if hyperparameters['batch_norm']:
        conv9 = BatchNormalization()(conv9)
    conv9 = Activation('relu')(conv9)
    conv9 = Conv2D(hyperparameters['base'], (3, 3), padding='same')(conv9)
    if hyperparameters['batch_norm']:
        conv9 = BatchNormalization()(conv9)
    conv9 = Activation('relu')(conv9)
    conv10 = Conv2D(1, (1, 1), activation='sigmoid')(conv9)
    model = Model(inputs=[inputs], outputs=[conv10])
    print(model.summary())
    model.compile(loss=hyperparameters['loss'],
                  optimizer=hyperparameters['optimizer'](lr=hyperparameters['lr']),
                  metrics=hyperparameters['metrics_func'])
    return model


def plot_history(hyperparameters, History, task_number):

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
    for metric in hyperparameters['metrics']:
        for key in History.history:
            if "val" not in key and metric in key:
                accuracy = key
            if "val" in key and metric in key:
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
            result_path = os.path.join(os.path.join(os.getcwd(), 'results'), str(task_number) + '_' + metric + '_accuracy.png')
            fig.savefig(result_path, dpi=fig.dpi)


