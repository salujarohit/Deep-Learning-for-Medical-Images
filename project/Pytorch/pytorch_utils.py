import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision

import matplotlib.pyplot as plt
import json
import os


def fix_rescale(rescale_str):
    rescale_list = rescale_str.split('/')
    if len(rescale_list) > 1:
        return float(rescale_list[0]) / float(rescale_list[1])
    else:
        return float(rescale_str)


def process_task_parameters(task_parameters):
    if task_parameters.get('train_parameters'):
        train_parameters = task_parameters['train_parameters']
    
        str_func_dict = {'dice_coef': dice_coef, 'precision': precision, 'recall': recall, 'f1': f1,
                         'SGD': optim.SGD, 'Adam': optim.Adam, 'dice_loss': dice_coef_loss, 'weighted_loss': weighted_loss,
                         'competition_dice_coef': competition_dice_coef, 'competition_dice_loss': competition_dice_loss, 'cross_entopy': nn.CrossEntropyLoss}
        if train_parameters['optimizer'] in str_func_dict:
            train_parameters['optimizer'] = str_func_dict[train_parameters['optimizer']]
    
        train_parameters['metrics_func'] = []
        for i, metric in enumerate(train_parameters['metrics']):
            if metric in str_func_dict:
                train_parameters['metrics_func'].append(str_func_dict[metric])
            else:
                train_parameters['metrics_func'].append(metric)
    
        if train_parameters['loss'] in str_func_dict:
            train_parameters['loss'] = str_func_dict[train_parameters['loss']]
    
        if train_parameters.get('generator') and train_parameters['generator'].get('rescale'):
            train_parameters['generator']['rescale'] = fix_rescale(train_parameters['generator']['rescale'])

        if train_parameters.get('last_layer_activation') == 'sigmoid':
            train_parameters['last_layer_activation'] = torch.sigmoid
        elif train_parameters.get('last_layer_activation') == 'softmax':
            train_parameters['last_layer_activation'] = torch.softmax

        task_parameters['train_parameters'] = train_parameters
    return task_parameters


def weighted_loss(weight_map, weight_strength):
    def weighted_dice_loss(y_true, y_pred):
        weight_f = torch.flatten(weight_map)
        weight_f = weight_f * weight_strength
        weight_f = 1 / (weight_f + 1)
        y_true_f = torch.flatten(y_true)
        y_true_f = y_true_f * weight_f
        y_pred_f = torch.flatten(y_pred)
        y_pred_f = y_pred_f * weight_f
        return 1 - dice_coef(y_true_f, y_pred_f)
    return weighted_dice_loss


def dice_coef_loss(y_true, y_pred):
    return 1-dice_coef(y_true, y_pred)


def dice_coef(y_true, y_pred, smooth=1):
    y_true_f = torch.flatten(y_true)
    y_pred_f = torch.flatten(y_pred)
    intersection = torch.sum(torch.abs(y_true_f * y_pred_f))
    return (2. * intersection + smooth) / (torch.sum(y_true_f) + torch.sum(y_pred_f) + smooth)


def competition_dice_loss(y_true, y_pred):
    return 1-competition_dice_coef(y_true, y_pred)


# def competition_dice_coef(y_true, y_pred, smooth=1):
#     y_pred = K.argmax(y_pred, axis=-1)
#     y_true = K.argmax(y_true, axis=-1)
#     # try:
#     # Compute tumor+kidney Dice
#     tk_pd = K.greater(y_pred, 0)
#     tk_gt = K.greater(y_true, 0)
#     intersection = K.all(K.stack([tk_gt, tk_pd], axis=3), axis=3)
#     tk_dice = (2 * K.sum(K.cast(intersection, K.floatx())) + smooth)/ (
#             K.sum(K.cast(tk_pd, K.floatx())) + K.sum(K.cast(tk_gt, K.floatx())) + smooth
#     )
#     # except ZeroDivisionError:
#     #     return 0
#
#     # try:
#         # Compute tumor Dice
#     tu_pd = K.greater(y_pred, 1)
#     tu_gt = K.greater(y_true, 1)
#     intersection = K.all(K.stack([tu_pd, tu_gt], axis=3), axis=3)
#     tu_dice = (2 * K.sum(K.cast(intersection, K.floatx())) + smooth)/ (
#             K.sum(K.cast(tu_pd, K.floatx())) + K.sum(K.cast(tu_gt, K.floatx())) + smooth
#     )
#     # except ZeroDivisionError:
#     #     return tk_dice / 2.0
#     return (tk_dice+tu_dice) / 2.0

def competition_dice_coef(y_true, y_pred, smooth=1):
    # try:
    # Compute tumor+kidney Dice
    tk_pd = torch.flatten(y_pred[:,:,:,1:3])
    tk_gt = torch.flatten(y_true[:,:,:,1:3])
    intersection = torch.sum(torch.abs(tk_gt * tk_pd))
    tk_dice = (2. * intersection + smooth) / (
            torch.sum(tk_pd) + torch.sum(tk_gt) + smooth
    )
    # except ZeroDivisionError:
    #     return 0

    # try:
        # Compute tumor Dice
    tu_pd = torch.flatten(y_pred[:,:,:,2:3])
    tu_gt = torch.flatten(y_true[:,:,:,2:3])
    intersection = torch.sum(torch.abs(tu_gt * tu_pd))
    tu_dice = (2. * intersection + smooth) / (
            torch.sum(tu_pd) + torch.sum(tu_gt) + smooth
    )
    # except ZeroDivisionError:
    #     return tk_dice / 2.0
    return (tk_dice+tu_dice) / 2.0


def recall(y_true, y_pred):
    y_true_f = torch.flatten(y_true)
    y_pred_f = torch.flatten(y_pred)
    true_positives = torch.sum(torch.round(torch.clmp(y_true_f * y_pred_f, 0, 1)))
    possible_positives = torch.sum(torch.round(torch.clip(y_true_f, 0, 1)))
    recall_result = true_positives / (possible_positives + 1e-7)
    return recall_result


def precision(y_true, y_pred):
    y_true_f = torch.flatten(y_true)
    y_pred_f = torch.flatten(y_pred)
    true_positives = torch.sum(torch.round(torch.clmp(y_true_f * y_pred_f, 0, 1)))
    predicted_positives = torch.sum(torch.round(torch.clmp(y_pred_f, 0, 1)))
    precision_result = true_positives / (predicted_positives + 1e-7)
    return precision_result


def f1(y_true, y_pred):
    precision_value = precision(y_true, y_pred)
    recall_value = recall(y_true, y_pred)
    return 2 * ((precision_value * recall_value) / (precision_value + recall_value + 1e-7))


def plot_pair(img_tensor,mask_tensor):
    fig, ax = plt.subplots(1, 2, figsize=(14, 2))
    ax[0].imshow(img_tensor.transpose((1, 2, 0).numpy()), cmap="gray")
    ax[1].imshow(mask_tensor.transpose((1, 2, 0).numpy()), cmap="gray")
    plt.show()


def plot_triplet(img_tensor, weight_tensor, mask_tensor):
    fig, ax = plt.subplots(1, 3, figsize=(14, 2))
    ax[0].imshow(img_tensor.transpose((1, 2, 0).numpy()), cmap="gray")
    ax[1].imshow(weight_tensor.transpose((1, 2, 0).numpy()), cmap="gray")
    ax[2].imshow(mask_tensor.transpose((1, 2, 0).numpy()), cmap="gray")
    plt.show()


def plit_grid(images_tensor):
    grid = torchvision.utils.make_grid(images_tensor)
    plt.imshow(grid.numpy().transpose((1, 2, 0)))
    plt.show()


def get_task_parameters(task):
    file_name = 'tasks/'+task+'.json'
    file_path = os.path.join(os.getcwd(), file_name)
    with open(file_path) as file:
        task_parameters = json.load(file)
        return process_task_parameters(task_parameters)



