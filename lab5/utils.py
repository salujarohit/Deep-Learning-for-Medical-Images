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


def hyperparameters_processing(task, hyperparameters):

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

    if 'generator' in hyperparameters:
        hyperparameters['generator']['rescale'] = fix_rescale(hyperparameters['generator']['rescale'])
    if 'test_generator' in hyperparameters:
        hyperparameters['test_generator']['rescale'] = fix_rescale(hyperparameters['test_generator']['rescale'])

    if "test_size" in hyperparameters:
        hyperparameters['num_folds'] = 1
    if 'autocontext_step' not in hyperparameters:
        hyperparameters['autocontext_step'] = 1
    if task == "1a":
        hyperparameters['batch_shape'] = (hyperparameters['batch_size'], hyperparameters['time_steps'], 1)
    elif task == "2a":
        hyperparameters['input_shape'] = (None, hyperparameters['time_steps'])
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


def get_hyperparameters(task):
    file_name = 'tasks/'+task+'.json'
    file_path = os.path.join(os.getcwd(), file_name)
    with open(file_path) as file:
        return json.load(file)


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

