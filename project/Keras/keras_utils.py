try:
    from tensorflow.keras.optimizers import Adam, SGD, RMSprop
    import tensorflow.keras.backend as K
except:
    from tensorflow.python.keras.optimizers import Adam, SGD, RMSprop
    import tensorflow.python.keras.backend as K
import tensorflow as tf
from skimage.transform import resize
import numpy as np
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
    for key in task_parameters:
        parameters = task_parameters[key]
        str_func_dict = {'dice_coef': dice_coef, 'precision': precision, 'recall': recall, 'f1': f1, 'RMSprop': RMSprop,
                         'SGD': SGD, 'Adam': Adam, 'dice_loss': dice_coef_loss, 'weighted_loss': weighted_loss,
                         'competition_coef': competition_coef, 'custom_competition_coef': custom_competition_coef,
                         'competition_loss': competition_loss, 'weighted_competition_coef':weighted_competition_coef,
                         "weighted_competition_loss": weighted_competition_loss}
        if parameters.get('optimizer') and  parameters['optimizer'] in str_func_dict:
            parameters['optimizer'] = str_func_dict[parameters['optimizer']]
    
        parameters['metrics_func'] = []
        if parameters.get('metrics'):
            for i, metric in enumerate(parameters.get('metrics')):
                if metric in str_func_dict:
                    parameters['metrics_func'].append(str_func_dict[metric])
                else:
                    parameters['metrics_func'].append(metric)
    
        if parameters.get('loss') and parameters['loss'] in str_func_dict:
            parameters['loss_func'] = str_func_dict[parameters['loss']]
    
        if parameters.get('generator') and parameters['generator'].get('rescale'):
            parameters['generator']['rescale'] = fix_rescale(parameters['generator']['rescale'])
        task_parameters[key] = parameters

    return task_parameters


def weighted_loss(weight_map, weight_strength):
    def weighted_dice_loss(y_true, y_pred):
        weight_f = K.flatten(weight_map)
        weight_f = weight_f * weight_strength
        weight_f = 1 / (weight_f + 1)
        y_true_f = K.flatten(y_true)
        y_true_f = y_true_f * weight_f
        y_pred_f = K.flatten(y_pred)
        y_pred_f = y_pred_f * weight_f
        return 1 - dice_coef(y_true_f, y_pred_f)
    return weighted_dice_loss


def dice_coef_loss(y_true, y_pred):
    return 1-dice_coef(y_true, y_pred)


def dice_coef(y_true, y_pred, smooth=1):
    y_true_f = K.flatten(y_true)
    y_pred_f = K.flatten(y_pred)
    intersection = K.sum(K.abs(y_true_f * y_pred_f))
    return (2. * intersection + smooth) / (K.sum(y_true_f) + K.sum(y_pred_f) + smooth)


def competition_loss(y_true, y_pred):
    return 1-custom_competition_coef(y_true, y_pred)


def competition_coef(y_true, y_pred, smooth=1):
    y_pred = K.argmax(y_pred, axis=-1)
    y_true = K.argmax(y_true, axis=-1)
    # try:
    # Compute tumor+kidney Dice
    tk_pd = K.greater(y_pred, 0)
    tk_gt = K.greater(y_true, 0)
    intersection = K.all(K.stack([tk_gt, tk_pd], axis=3), axis=3)
    tk_dice = (2 * K.sum(K.cast(intersection, K.floatx())) + smooth)/ (
            K.sum(K.cast(tk_pd, K.floatx())) + K.sum(K.cast(tk_gt, K.floatx())) + smooth
    )
    # except ZeroDivisionError:
    #     return 0

    # try:
        # Compute tumor Dice
    tu_pd = K.greater(y_pred, 1)
    tu_gt = K.greater(y_true, 1)
    intersection = K.all(K.stack([tu_pd, tu_gt], axis=3), axis=3)
    tu_dice = (2 * K.sum(K.cast(intersection, K.floatx())) + smooth)/ (
            K.sum(K.cast(tu_pd, K.floatx())) + K.sum(K.cast(tu_gt, K.floatx())) + smooth
    )
    # except ZeroDivisionError:
    #     return tk_dice / 2.0
    return (tk_dice+tu_dice) / 2.0


def custom_competition_coef(y_true, y_pred, smooth=1):
    # try:
    # Compute tumor+kidney Dice
    tk_pd = K.flatten(y_pred[:,:,:,1:3])
    tk_gt = K.flatten(y_true[:,:,:,1:3])
    intersection = K.sum(K.abs(tk_gt * tk_pd))
    tk_dice = (2. * intersection + smooth) / (
            K.sum(tk_pd) + K.sum(tk_gt) + smooth
    )
    # except ZeroDivisionError:
    #     return 0

    # try:
        # Compute tumor Dice
    tu_pd = K.flatten(y_pred[:,:,:,2:3])
    tu_gt = K.flatten(y_true[:,:,:,2:3])
    intersection = K.sum(K.abs(tu_gt * tu_pd))
    tu_dice = (2. * intersection + smooth) / (
            K.sum(tu_pd) + K.sum(tu_gt) + smooth
    )
    # except ZeroDivisionError:
    #     return tk_dice / 2.0
    return (tk_dice+tu_dice) / 2.0


def weighted_competition_loss(y_true, y_pred):
    return 1-weighted_competition_coef(y_true, y_pred)


def weighted_competition_coef(y_true, y_pred, smooth=1):
    # try:
    # Compute tumor+kidney Dice
    tk_pd = K.flatten(y_pred[:,:,:,1:3])
    tk_gt = K.flatten(y_true[:,:,:,1:3])
    intersection = K.sum(K.abs(tk_gt * tk_pd))
    tk_dice = (2. * intersection + smooth) / (
            K.sum(tk_pd) + K.sum(tk_gt) + smooth
    )
    # except ZeroDivisionError:
    #     return 0

    # try:
        # Compute tumor Dice
    tu_pd = K.flatten(y_pred[:,:,:,2:3])
    tu_gt = K.flatten(y_true[:,:,:,2:3])
    intersection = K.sum(K.abs(tu_gt * tu_pd))
    tu_dice = (2. * intersection + smooth) / (
            K.sum(tu_pd) + K.sum(tu_gt) + smooth
    )
    # except ZeroDivisionError:
    #     return tk_dice / 2.0
    return (.1 * tk_dice) + (.9 * tu_dice)

def recall(y_true, y_pred):
    y_true_f = K.flatten(y_true)
    y_pred_f = K.flatten(y_pred)
    true_positives = K.sum(K.round(K.clip(y_true_f * y_pred_f, 0, 1)))
    possible_positives = K.sum(K.round(K.clip(y_true_f, 0, 1)))
    recall_result = true_positives / (possible_positives + K.epsilon())
    return recall_result


def precision(y_true, y_pred):
    y_true_f = K.flatten(y_true)
    y_pred_f = K.flatten(y_pred)
    true_positives = K.sum(K.round(K.clip(y_true_f * y_pred_f, 0, 1)))
    predicted_positives = K.sum(K.round(K.clip(y_pred_f, 0, 1)))
    precision_result = true_positives / (predicted_positives + K.epsilon())
    return precision_result


def f1(y_true, y_pred):
    precision_value = precision(y_true, y_pred)
    recall_value = recall(y_true, y_pred)
    return 2 * ((precision_value * recall_value) / (precision_value + recall_value + K.epsilon()))


def plot_pair(img,mask):
    fig, ax = plt.subplots(1, 2, figsize=(14, 2))
    ax[0].imshow(img,cmap="gray")
    ax[1].imshow(mask, cmap="gray")
    plt.show()


def plot_triplet(img, weight, mask):
    fig, ax = plt.subplots(1, 3, figsize=(14, 2))
    ax[0].imshow(img, cmap="gray")
    ax[1].imshow(weight, cmap="gray")
    ax[2].imshow(mask, cmap="gray")
    plt.show()


def get_task_parameters(task):
    file_name = 'tasks/'+task+'.json'
    file_path = os.path.join(os.getcwd(), file_name)
    with open(file_path) as file:
        task_parameters = json.load(file)
        return process_task_parameters(task_parameters)


def update_board (hyperparameters, evaluation, task):
    file_name = 'tasks/' + 'board' + '.json'
    file_path = os.path.join(os.getcwd(), file_name)
    with open(file_path) as file:
        board_dict = json.load(file)
    if task not in board_dict:
        board_dict[task] = {}
    if task in board_dict:
        board_dict[task]['loss'] = str(evaluation[0])
        for metric in hyperparameters['metrics']:
            board_dict[task][metric] = str(evaluation[1])
    with open(file_path, 'w') as outfile:
        json.dump(board_dict, outfile)


def save_step_prediction(predictions, step_num):
    save_path = os.path.join(os.getcwd(), 'models')
    if not os.path.isdir(save_path):
        os.mkdir(save_path)
    np.save(os.path.join(save_path, 'posterior_unet_step' + str(step_num) + '.npy'), predictions)
