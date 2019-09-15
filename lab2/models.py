import numpy as np
import matplotlib.pyplot as plt
import os
try:
    from tensorflow.keras.models import Sequential
    from tensorflow.keras.layers import Flatten, Conv2D, MaxPooling2D, Activation, Dropout, Dense, BatchNormalization, SpatialDropout2D, GlobalAveragePooling2D
    from tensorflow.keras.optimizers import Adam, SGD, RMSprop
    from tensorflow.keras.applications import VGG16, InceptionV3

except:
    import tensorflow as tf
    from tensorflow.python.keras.models import Sequential
    from tensorflow.python.keras.layers.core import Flatten, Conv2D, MaxPooling2D, Activation, Dropout, Dense, BatchNormalization, SpatialDropout2D, GlobalAveragePooling2D
    from tensorflow.python.keras.optimizers import Adam, SGD, RMSprop
    from tensorflow.python.keras.applications import VGG16, InceptionV3

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
    model.add(Conv2D(hyperparameters['base']*2, (3, 3), padding='same'))
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
        return  get_pretrained_model(hyperparameters)
    elif hyperparameters['model'] == "alexnet":
        return model_AlexNet(hyperparameters)
    elif hyperparameters['model'] == "vgg16":
        return model_vgg16(hyperparameters)


def plot_history(History, task_number):
    fig = plt.figure(figsize=(4, 4))
    plt.title("Learning curve")
    plt.plot(History.history["loss"], label="loss")
    plt.plot(History.history["val_loss"], label="val_loss")
    plt.plot( np.argmin(History.history["val_loss"]),
             np.min(History.history["val_loss"]),
             marker="x", color="r", label="best model")

    plt.xlabel("Epochs")
    plt.ylabel("Loss Value")
    plt.legend()
    result_path = os.path.join(os.path.join(os.getcwd(), 'results'), str(task_number)+'_loss.png')
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
