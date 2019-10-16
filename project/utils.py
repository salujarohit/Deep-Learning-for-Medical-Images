from pathlib import Path

import nibabel as nib
import os
#
# def get_full_case_id(cid):
#     try:
#         cid = int(cid)
#         case_id = "case_{:05d}".format(cid)
#     except ValueError:
#         case_id = cid
#
#     return case_id
#
#
# def get_case_path(cid):
#     # Resolve location where data should be living
#     data_path =  Path("/project_data")
#     if not data_path.exists():
#         raise IOError(
#             "Data path, {}, could not be resolved".format(str(data_path))
#         )
#
#     # Get case_id from provided cid
#     case_id = get_full_case_id(cid)
#
#     # Make sure that case_id exists under the data_path
#     case_path = data_path / case_id
#     if not case_path.exists():
#         raise ValueError(
#             "Case could not be found \"{}\"".format(case_path.name)
#         )
#
#     return case_path
#
#
# def load_volume(cid):
#     case_path = get_case_path(cid)
#     vol = nib.load(str(case_path / "imaging.nii.gz"))
#     return vol
#
#
# def load_segmentation(cid):
#     case_path = get_case_path(cid)
#     seg = nib.load(str(case_path / "segmentation.nii.gz"))
#     return seg
#
#
# def load_case(cid):
#     vol = load_volume(cid)
#     seg = load_segmentation(cid)
#     return vol, seg
#
# def load_case(cid):
#     data_path = '/project_data_interpolated/'
#     img_name =''
#     mask_name = ''
#     for file in os.listdir(data_path):
#         if str(cid) in file and 'imaging' in file:
#             img_name = file
#         elif str(cid) in file and 'segmentation' in file:
#             mask_name = file
#     img = nib.load(os.path.join(data_path, img_name))
#     mask = nib.load(os.path.join(data_path, mask_name))
#     return img, mask

try:
    from tensorflow.keras.optimizers import Adam, SGD, RMSprop
    import tensorflow.keras.backend as K
except:
    from tensorflow.python.keras.optimizers import Adam, SGD, RMSprop
    import tensorflow.python.keras.backend as K
import matplotlib.pyplot as plt
import json
import os

def fix_rescale(rescale_str):
    rescale_list = rescale_str.split('/')
    if len(rescale_list) > 1:
        return float(rescale_list[0]) / float(rescale_list[1])
    else:
        return float(rescale_str)


def hyperparameters_processing(hyperparameters):

    str_func_dict = {'dice_coef': dice_coef, 'precision': precision, 'recall': recall, 'f1': f1, 'RMSprop': RMSprop,
                     'SGD': SGD, 'Adam': Adam, 'dice_loss': dice_coef_loss, 'weighted_loss': weighted_loss}
    if hyperparameters['optimizer'] in str_func_dict:
        hyperparameters['optimizer'] = str_func_dict[hyperparameters['optimizer']]

    hyperparameters['metrics_func'] = []
    for i, metric in enumerate(hyperparameters['metrics']):
        if metric in str_func_dict:
            hyperparameters['metrics_func'].append(str_func_dict[metric])
        else:
            hyperparameters['metrics_func'].append(metric)

    if hyperparameters['loss'] in str_func_dict:
        hyperparameters['loss'] = str_func_dict[hyperparameters['loss']]

    if hyperparameters.get('generator') and hyperparameters['generator'].get('rescale'):
        hyperparameters['generator']['rescale'] = fix_rescale(hyperparameters['generator']['rescale'])

    return hyperparameters


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


def get_hyperparameters(task):
    file_name = 'tasks/'+task+'.json'
    file_path = os.path.join(os.getcwd(), file_name)
    with open(file_path) as file:
        hyperparameters = json.load(file)
        return hyperparameters_processing(hyperparameters)


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

