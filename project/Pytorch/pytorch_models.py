import numpy as np
import matplotlib.pyplot as plt
import os

import torch
import torch.nn as nn
import torch.nn.functional as F
from collections import OrderedDict


class UNet(nn.Module):

    def __init__(self, in_channels=3, out_channels=1, init_features=16, out_activation= torch.sigmoid):
        super(UNet, self).__init__()
        self.out_activation = out_activation
        features = init_features
        self.encoder1 = UNet._block(in_channels, features, name="enc1")
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.encoder2 = UNet._block(features, features * 2, name="enc2")
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.encoder3 = UNet._block(features * 2, features * 4, name="enc3")
        self.pool3 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.encoder4 = UNet._block(features * 4, features * 8, name="enc4")
        self.pool4 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.bottleneck = UNet._block(features * 8, features * 16, name="bottleneck")

        self.upconv4 = nn.ConvTranspose2d(
            features * 16, features * 8, kernel_size=2, stride=2
        )
        self.decoder4 = UNet._block((features * 8) * 2, features * 8, name="dec4")
        self.upconv3 = nn.ConvTranspose2d(
            features * 8, features * 4, kernel_size=2, stride=2
        )
        self.decoder3 = UNet._block((features * 4) * 2, features * 4, name="dec3")
        self.upconv2 = nn.ConvTranspose2d(
            features * 4, features * 2, kernel_size=2, stride=2
        )
        self.decoder2 = UNet._block((features * 2) * 2, features * 2, name="dec2")
        self.upconv1 = nn.ConvTranspose2d(
            features * 2, features, kernel_size=2, stride=2
        )
        self.decoder1 = UNet._block(features * 2, features, name="dec1")

        self.conv = nn.Conv2d(
            in_channels=features, out_channels=out_channels, kernel_size=1
        )

    def forward(self, x):
        enc1 = self.encoder1(x)
        enc2 = self.encoder2(self.pool1(enc1))
        enc3 = self.encoder3(self.pool2(enc2))
        enc4 = self.encoder4(self.pool3(enc3))

        bottleneck = self.bottleneck(self.pool4(enc4))

        dec4 = self.upconv4(bottleneck)
        dec4 = torch.cat((dec4, enc4), dim=1)
        dec4 = self.decoder4(dec4)
        dec3 = self.upconv3(dec4)
        dec3 = torch.cat((dec3, enc3), dim=1)
        dec3 = self.decoder3(dec3)
        dec2 = self.upconv2(dec3)
        dec2 = torch.cat((dec2, enc2), dim=1)
        dec2 = self.decoder2(dec2)
        dec1 = self.upconv1(dec2)
        dec1 = torch.cat((dec1, enc1), dim=1)
        dec1 = self.decoder1(dec1)
        return self.out_activation(self.conv(dec1))

    @staticmethod
    def _block(in_channels, features, name):
        return nn.Sequential(
            OrderedDict(
                [
                    (
                        name + "conv1",
                        nn.Conv2d(
                            in_channels=in_channels,
                            out_channels=features,
                            kernel_size=3,
                            padding=1,
                            bias=False,
                        ),
                    ),
                    (name + "norm1", nn.BatchNorm2d(num_features=features)),
                    (name + "relu1", nn.ReLU(inplace=True)),
                    (
                        name + "conv2",
                        nn.Conv2d(
                            in_channels=features,
                            out_channels=features,
                            kernel_size=3,
                            padding=1,
                            bias=False,
                        ),
                    ),
                    (name + "norm2", nn.BatchNorm2d(num_features=features)),
                    (name + "relu2", nn.ReLU(inplace=True)),
                ]
            )
        )


def get_unet(hyperparameters):
    in_channels = 2 if hyperparameters.get('autocontext_step') else hyperparameters['input_shape'][2]
    model = UNet(in_channels = in_channels, out_channels = hyperparameters['last_layer_units'],
                 init_features = hyperparameters['base'], out_activation=hyperparameters['last_layer_activation'])

    # if hyperparameters.get('use_weight_maps'):
    #     weight_input = Input(hyperparameters['input_shape'])
    #     model = Model(inputs=[inputs, weight_input], outputs=[conv10])
    #     loss = hyperparameters['loss'](weight_input, hyperparameters['weight_strength'])
    # else:
    #     model = Model(inputs=[inputs], outputs=[conv10])
    #     loss = hyperparameters['loss']
    return model


def plot_loss(task_number, training_loss, val_loss, step_num=None, fold_num=None):
    if not os.path.isdir(os.path.join(os.getcwd(), 'results')):
        os.mkdir(os.path.join(os.getcwd(), 'results'))
    task_path = str(task_number)
    if step_num is not None:
        task_path += '_step' + str(step_num)
    if fold_num is not None:
        task_path += '_fold' + str(fold_num)

    fig = plt.figure(figsize=(4, 4))
    plt.title("Learning Curve")
    plt.plot(training_loss, label="loss")
    plt.plot(val_loss, label="val_loss")
    plt.plot(np.argmin(val_loss),
             np.min(val_loss),
             marker="x", color="r", label="best model")

    plt.xlabel("Epochs")
    plt.ylabel("Loss Value")
    plt.legend()
    result_path = os.path.join(os.path.join(os.getcwd(), 'results'),  task_path + '_loss.png')
    fig.savefig(result_path, dpi=fig.dpi)


def plot_history(hyperparameters, history, task_number, step_num =None, fold_num=None):
    if not os.path.isdir(os.path.join(os.getcwd(), 'results')):
        os.mkdir(os.path.join(os.getcwd(), 'results'))
    task_path = str(task_number)
    if step_num is not None:
        task_path += '_step' + str(step_num)
    if fold_num is not None:
        task_path += '_fold' + str(fold_num)

    fig = plt.figure(figsize=(4, 4))
    plt.title("Learning Curve")
    plt.plot(history.history["loss"], label="loss")
    plt.plot(history.history["val_loss"], label="val_loss")
    plt.plot(np.argmin(history.history["val_loss"]),
             np.min(history.history["val_loss"]),
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
    for metric in hyperparameters['metrics']:
        for key in history.history:
            if "val" not in key and metric in key:
                metric_key = key
            if "val" in key and metric in key:
                metric_val_key = key
        if metric_key != '' and metric_val_key != '':
            plt.plot(history.history[metric_key], label=metric_key)
            plt.plot(history.history[metric_val_key], label=metric_val_key)
            # plt.plot(np.argmax(history.history[metric_val_key]),
            #          np.max(history.history[metric_val_key]),
            #          marker="x", color="r", label="best model")

    plt.xlabel("Epochs")
    plt.ylabel("Metrics Value")
    plt.legend()
    result_path = os.path.join(os.path.join(os.getcwd(), 'results'), task_path + '_metrics.png')
    fig.savefig(result_path, dpi=fig.dpi)


def save_model(model, task_number, step_num=None, fold_num=None):
    if not os.path.isdir(os.path.join(os.getcwd(), 'models')):
        os.mkdir(os.path.join(os.getcwd(), 'models'))

    task_path = str(task_number)
    if step_num is not None:
        task_path += '_step' + str(step_num)
    if fold_num is not None:
        task_path += '_fold' + str(fold_num)
    model_path = os.path.join(os.path.join(os.getcwd(), 'models'), task_path + '.h5')
    model.save(model_path)


def save_step_prediction(predictions, step_num):
    save_path = os.path.join(os.getcwd(), 'models')
    if not os.path.isdir(save_path):
        os.mkdir(save_path)
    np.save(os.path.join(save_path, 'posterior_unet_step' + str(step_num) + '.npy'), predictions)


