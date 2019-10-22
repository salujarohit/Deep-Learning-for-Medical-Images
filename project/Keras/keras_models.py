import numpy as np
import matplotlib.pyplot as plt
import os
from keras_utils import save_step_prediction
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
    from tensorflow.keras.models import load_model
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
    from tensorflow.python.keras.models import load_model
from skimage.transform import resize
import nibabel as nib

class ModelContainer:
    def __init__(self, hyperparameters, current_task):
        self.hyperparameters = hyperparameters
        self.model = None
        self.model_history = None
        self.current_task = current_task
        self.step_num = None
        self.fold_num = None
        self.data_loader = None
        self.predict_loader = None
        self.get_model()

    def get_unet(self):
        input_shape = (self.hyperparameters['input_shape'][0], self.hyperparameters['input_shape'][1], 2) if self.hyperparameters.get('autocontext_step') \
            else self.hyperparameters['input_shape']
        inputs = Input(input_shape)
        conv1 = Conv2D(self.hyperparameters['base'], (3, 3), padding='same')(inputs)
        if self.hyperparameters['batch_norm']:
            conv1 = BatchNormalization()(conv1)
        conv1 = Activation('relu')(conv1)
        conv1 = Conv2D(self.hyperparameters['base'], (3, 3), padding='same')(conv1)
        if self.hyperparameters['batch_norm']:
            conv1 = BatchNormalization()(conv1)
        conv1 = Activation('relu')(conv1)
        pool1 = MaxPooling2D(pool_size=(2, 2))(conv1)
        if self.hyperparameters['dropout'] != 0:
            pool1 = Dropout(self.hyperparameters['dropout'])(pool1)
        conv2 = Conv2D(self.hyperparameters['base'] * 2, (3, 3), padding='same')(pool1)
        if self.hyperparameters['batch_norm']:
            conv2 = BatchNormalization()(conv2)
        conv2 = Activation('relu')(conv2)
        conv2 = Conv2D(self.hyperparameters['base'] * 2, (3, 3), padding='same')(conv2)
        if self.hyperparameters['batch_norm']:
            conv2 = BatchNormalization()(conv2)
        conv2 = Activation('relu')(conv2)
        pool2 = MaxPooling2D(pool_size=(2, 2))(conv2)
        if self.hyperparameters['dropout'] != 0:
            pool2 = Dropout(self.hyperparameters['dropout'])(pool2)
        conv3 = Conv2D(self.hyperparameters['base'] * 4, (3, 3), padding='same')(pool2)
        if self.hyperparameters['batch_norm']:
            conv3 = BatchNormalization()(conv3)
        conv3 = Activation('relu')(conv3)
        conv3 = Conv2D(self.hyperparameters['base'] * 4, (3, 3), padding='same')(conv3)
        if self.hyperparameters['batch_norm']:
            conv3 = BatchNormalization()(conv3)
        conv3 = Activation('relu')(conv3)
        pool3 = MaxPooling2D(pool_size=(2, 2))(conv3)
        if self.hyperparameters['dropout'] != 0:
            pool3 = Dropout(self.hyperparameters['dropout'])(pool3)
        conv4 = Conv2D(self.hyperparameters['base'] * 8, (3, 3), padding='same')(pool3)
        if self.hyperparameters['batch_norm']:
            conv4 = BatchNormalization()(conv4)
        conv4 = Activation('relu')(conv4)
        conv4 = Conv2D(self.hyperparameters['base'] * 8, (3, 3), padding='same')(conv4)
        if self.hyperparameters['batch_norm']:
            conv4 = BatchNormalization()(conv4)
        conv4 = Activation('relu')(conv4)
        pool4 = MaxPooling2D(pool_size=(2, 2))(conv4)
        if self.hyperparameters['dropout'] != 0:
            pool4 = Dropout(self.hyperparameters['dropout'])(pool4)
        conv5 = Conv2D(self.hyperparameters['base'] * 16, (3, 3), padding='same')(pool4)
        if self.hyperparameters['batch_norm']:
            conv5 = BatchNormalization()(conv5)
        conv5 = Activation('relu')(conv5)
        conv5 = Conv2D(self.hyperparameters['base'] * 16, (3, 3), padding='same')(conv5)
        if self.hyperparameters['batch_norm']:
            conv5 = BatchNormalization()(conv5)
        conv5 = Activation('relu')(conv5)
        up6 = concatenate([Conv2DTranspose(self.hyperparameters['base'] * 8, (2, 2), strides=(2, 2), padding='same')(conv5), conv4], axis=3)
        conv6 = Conv2D(self.hyperparameters['base'] * 8, (3, 3), padding='same')(up6)
        if self.hyperparameters['batch_norm']:
            conv6 = BatchNormalization()(conv6)
        conv6 = Activation('relu')(conv6)
        conv6 = Conv2D(self.hyperparameters['base'] * 8, (3, 3), padding='same')(conv6)
        if self.hyperparameters['batch_norm']:
            conv6 = BatchNormalization()(conv6)
        conv6 = Activation('relu')(conv6)
        up7 = concatenate([Conv2DTranspose(128, (2, 2), strides=(2, 2), padding='same')(conv6), conv3], axis=3)
        conv7 = Conv2D(self.hyperparameters['base'] * 4, (3, 3), padding='same')(up7)
        if self.hyperparameters['batch_norm']:
            conv7 = BatchNormalization()(conv7)
        conv7 = Activation('relu')(conv7)
        conv7 = Conv2D(self.hyperparameters['base'] * 4, (3, 3), padding='same')(conv7)
        if self.hyperparameters['batch_norm']:
            conv7 = BatchNormalization()(conv7)
        conv7 = Activation('relu')(conv7)
        up8 = concatenate([Conv2DTranspose(self.hyperparameters['base'] * 2, (2, 2), strides=(2, 2), padding='same')(conv7), conv2], axis=3)
        conv8 = Conv2D(self.hyperparameters['base'] * 2, (3, 3), padding='same')(up8)
        if self.hyperparameters['batch_norm']:
            conv8 = BatchNormalization()(conv8)
        conv8 = Activation('relu')(conv8)
        conv8 = Conv2D(self.hyperparameters['base'] * 2, (3, 3), padding='same')(conv8)
        if self.hyperparameters['batch_norm']:
            conv8 = BatchNormalization()(conv8)
        conv8 = Activation('relu')(conv8)
        up9 = concatenate([Conv2DTranspose(self.hyperparameters['base'], (2, 2), strides=(2, 2), padding='same')(conv8), conv1], axis=3)
        conv9 = Conv2D(self.hyperparameters['base'], (3, 3), padding='same')(up9)
        if self.hyperparameters['batch_norm']:
            conv9 = BatchNormalization()(conv9)
        conv9 = Activation('relu')(conv9)
        conv9 = Conv2D(self.hyperparameters['base'], (3, 3), padding='same')(conv9)
        if self.hyperparameters['batch_norm']:
            conv9 = BatchNormalization()(conv9)
        conv9 = Activation('relu')(conv9)
        conv10 = Conv2D(self.hyperparameters['last_layer_units'], (1, 1), activation=self.hyperparameters['last_layer_activation'])(conv9)

        if self.hyperparameters.get('use_weight_maps'):
            weight_input = Input(self.hyperparameters['input_shape'])
            self.model = Model(inputs=[inputs, weight_input], outputs=[conv10])
            loss = self.hyperparameters['loss_func'](weight_input, self.hyperparameters['weight_strength'])
        else:
            self.model = Model(inputs=[inputs], outputs=[conv10])
            loss = self.hyperparameters['loss_func']

        print(self.model.summary())
        self.model.compile(loss=loss,
                      optimizer=self.hyperparameters['optimizer'](lr=self.hyperparameters['lr']),
                      metrics=self.hyperparameters['metrics_func'])
        return self.model

    def plot_history(self):
        if not os.path.isdir(os.path.join(os.getcwd(), 'results')):
            os.mkdir(os.path.join(os.getcwd(), 'results'))
        task_path = str(self.current_task)
        if self.step_num is not None:
            task_path += '_step' + str(self.step_num)
        if self.fold_num is not None:
            task_path += '_fold' + str(self.fold_num)

        fig = plt.figure(figsize=(4, 4))
        plt.title("Learning Curve")
        plt.plot(self.model_history.history["loss"], label="loss")
        plt.plot(self.model_history.history["val_loss"], label="val_loss")
        plt.plot(np.argmin(self.model_history.history["val_loss"]),
                 np.min(self.model_history.history["val_loss"]),
                 marker="x", color="r", label="best model")

        plt.xlabel("Epochs")
        plt.ylabel("Loss Value")
        plt.legend()
        result_path = os.path.join(os.path.join(os.getcwd(), 'results'),  task_path + '_loss.png')

        fig.savefig(result_path, dpi=fig.dpi)

        metric_key = ''
        metric_val_key = ''
        fig = plt.figure(figsize=(4, 4))
        plt.title("Metrics Curves")
        for metric in self.hyperparameters['metrics']:
            for key in self.model_history.history:
                if "val" not in key and metric in key:
                    metric_key = key
                if "val" in key and metric in key:
                    metric_val_key = key
            if metric_key != '' and metric_val_key != '':
                plt.plot(self.model_history.history[metric_key], label=metric_key)
                plt.plot(self.model_history.history[metric_val_key], label=metric_val_key)
                # plt.plot(np.argmax(history.history[metric_val_key]),
                #          np.max(history.history[metric_val_key]),
                #          marker="x", color="r", label="best model")

        plt.xlabel("Epochs")
        plt.ylabel("Metrics Value")
        plt.legend()
        result_path = os.path.join(os.path.join(os.getcwd(), 'results'), task_path + '_metrics.png')
        fig.savefig(result_path, dpi=fig.dpi)

    def save_model(self):
        if not os.path.isdir(os.path.join(os.getcwd(), 'models')):
            os.mkdir(os.path.join(os.getcwd(), 'models'))

        task_path = str(self.current_task)
        if self.step_num is not None:
            task_path += '_step' + str(self.step_num)
        if self.fold_num is not None:
            task_path += '_fold' + str(self.fold_num)
        model_path = os.path.join(os.path.join(os.getcwd(), 'models'), task_path + '.h5')
        self.model.save(model_path)

    def load_saved_model(self):
        model_path = os.path.join(os.path.join(os.getcwd(), 'models'), self.hyperparameters['use_model'])
        if not os.path.isfile(model_path):
            raise ValueError("Model name is not correct")

        custom_objects = {}
        custom_objects[self.hyperparameters['loss']] = self.hyperparameters['loss_func']
        for i, metric in enumerate(self.hyperparameters['metrics']):
            custom_objects[metric] = self.hyperparameters['metrics_func'][i]
        self.model = load_model(model_path, custom_objects=custom_objects)
        return self.model

    def normal_train(self):
        training_generator, validation_generator = self.data_loader.get_generators()
        # for batch_x, batch_y in training_generator:
        #     plot_pair(batch_x[0,:,:,0], batch_y[0,:,:,0])
        #     plot_pair(batch_x[0, :, :, 0], batch_y[0, :, :, 1])
        #     plot_pair(batch_x[0, :, :, 0], batch_y[0, :, :, 2])
        self.model_history = self.model.fit_generator(training_generator,
                                                 epochs=self.hyperparameters['epochs'],
                                                 validation_data=validation_generator)
        self.plot_history()
        if self.hyperparameters.get('save_model'):
            self.save_model()

    def kfold_train(self):
        for fold_num in range(self.hyperparameters['folds']):
            self.fold_num = fold_num
            training_generator, validation_generator = self.data_loader.get_generators(fold_num=self.fold_num, step_num=self.step_num)
            self.model_history = self.model.fit_generator(training_generator,
                                                     epochs=self.hyperparameters['epochs'],
                                                     validation_data=validation_generator)
            self.plot_history()
            if self.hyperparameters.get('save_model'):
                self.save_model()

    def autcontext_train(self):
        autocontext_step = self.hyperparameters['autocontext_step']
        model_predictions = [None] * len(
            os.listdir(os.path.join(os.getcwd(), os.path.join(self.hyperparameters['data_path'], 'Image'))))
        for step_num in range(0, autocontext_step):
            self.step_num = step_num
            for fold_num in range(self.hyperparameters['folds']):
                self.fold_num = fold_num
                training_generator, validation_generator = self.data_loader.get_generators(fold_num, step_num)

                self.model_history = self.model.fit_generator(training_generator,
                                                    epochs=self.hyperparameters['epochs'],
                                                    validation_data=validation_generator)
                self.plot_history()
                if self.hyperparameters.get('save_model'):
                    self.save_model()
                y_pred = self.model.predict(validation_generator)
                total_val = len(validation_generator.image_filenames)
                model_predictions[(fold_num * total_val):((fold_num + 1) * total_val)] = y_pred
            save_step_prediction(model_predictions, step_num)

    def train(self, dataloader):
        self.data_loader = dataloader
        if self.hyperparameters.get('autocontext_step'):
            self.autcontext_train()
        elif self.hyperparameters.get('folds'):
            self.kfold_train()
        else:
            self.normal_train()

    def predict(self, dataloader, start_pred, num_cases, data_path):
        self.predict_loader = dataloader
        for i in range(start_pred, start_pred+num_cases):
            test_batch, original_shape = self.predict_loader.__getitem__(i)
            predictions = self.model.predict_on_batch(test_batch)
            output_predictions = []
            for prediction in predictions:
                prediction = resize(prediction, (original_shape[0], original_shape[1]))
                output_predictions.append(np.argmax(prediction, axis=2))
            output_predictions = np.array(output_predictions)
            affine = np.array([[0., 0., -0.781624972820282, 0.],
                               [0., -0.781624972820282, 0., 0.],
                               [-3., 0., 0., 0.],
                               [0., 0., 0., 1.]])
            img = nib.Nifti1Image(output_predictions, affine)
            prediction_path = os.path.join(os.getcwd(),
                                           os.path.join(data_path, 'predictions'))
            if not os.path.isdir(prediction_path):
                os.mkdir(prediction_path)
            img.to_filename(os.path.join(prediction_path, 'prediction_' + str(i).zfill(5) + '.nii.gz'))

    def get_model(self):
        if self.model is not None:
            return self.model
        if self.hyperparameters.get('use_model'):
            return self.load_saved_model()
        return self.get_unet()
