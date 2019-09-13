import numpy as np
import matplotlib.pyplot as plt
import os
try:
    from tensorflow.keras.models import Sequential
    from tensorflow.keras.layers import Flatten, Conv2D, MaxPooling2D, Activation, Dropout, Dense, BatchNormalization, SpatialDropout2D
    from tensorflow.keras.optimizers import Adam, SGD, RMSprop
except:
    import tensorflow as tf
    from tensorflow.python.keras.models import Sequential
    from tensorflow.python.keras.layers.core import Flatten, Conv2D, MaxPooling2D, Activation, Dropout, Dense, BatchNormalization, SpatialDropout2D
    from tensorflow.python.keras.optimizers import Adam, SGD, RMSprop

# LeNet Model
def model_LeNet(hyperparameters):
    model = Sequential()
    model.add(Conv2D(hyperparameters['base'], kernel_size=(3, 3), activation='relu',
                     strides=1, padding='same',
                     input_shape=hyperparameters['input_shape']))
    model.add(MaxPooling2D(pool_size=(2, 2)))

    model.add(Conv2D(hyperparameters['base'] * 2, kernel_size=(3, 3), activation='relu',
                     strides=1, padding='same'))
    model.add(MaxPooling2D(pool_size=(2, 2)))

    model.add(Flatten())
    model.add(Dense(hyperparameters['base'] * 2, activation='relu'))
    model.add(Dense(1, activation='sigmoid'))
    print(model.summary())
    model.compile(loss=hyperparameters['loss'],
                        optimizer=hyperparameters['optimizer'](lr=hyperparameters['lr']),
                        metrics=['binary_accuracy'])

    return model


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
    model.add(Dense(64))
    model.add(Activation('relu'))
    if len(hyperparameters['dropout']) > 0:
        model.add(Dropout(hyperparameters['dropout'][0]))
    model.add(Dense(64))
    model.add(Activation('relu'))
    if len(hyperparameters['dropout']) > 1:
        model.add(Dropout(hyperparameters['dropout'][1]))
    model.add(Dense(1))
    model.add(Activation('sigmoid'))

    print(model.summary())
    model.compile(loss=hyperparameters['loss'],
                        optimizer=hyperparameters['optimizer'](lr=hyperparameters['lr']),
                        metrics=['binary_accuracy'])

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
    model.add(Dense(64))
    model.add(Activation('relu'))
    if len(hyperparameters['dropout']) > 0:
        model.add(Dropout(hyperparameters['dropout'][0]))
    model.add(Dense(64))
    model.add(Activation('relu'))
    if len(hyperparameters['dropout']) > 1:
        model.add(Dropout(hyperparameters['dropout'][1]))
    model.add(Dense(1, activation='sigmoid'))


    print(model.summary())
    model.compile(loss=hyperparameters['loss'],
                  optimizer=hyperparameters['optimizer'](lr=hyperparameters['lr']),
                  metrics=['binary_accuracy'])

    return model

def get_model(hyperparameters):
    if hyperparameters['model'] == "lenet":
        return model_LeNet(hyperparameters)
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
