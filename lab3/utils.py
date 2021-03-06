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

    if hyperparameters['optimizer'] == "RMSprop":
        hyperparameters['optimizer'] = RMSprop
    elif hyperparameters['optimizer'] == "SGD":
        hyperparameters['optimizer'] = SGD
    else:
        hyperparameters['optimizer'] = Adam

    hyperparameters['metrics_func'] = []
    for i, metric in enumerate(hyperparameters['metrics']):
        if metric == 'dice_coef':
            hyperparameters['metrics_func'].append(dice_coef)
        elif metric == 'precision':
            hyperparameters['metrics_func'].append(precision)
        elif metric == 'recall':
            hyperparameters['metrics_func'].append(recall)
        elif metric == 'f1':
            hyperparameters['metrics_func'].append(f1)
        else:
            hyperparameters['metrics_func'].append(metric)

    if hyperparameters['loss'] == 'dice_loss':
        hyperparameters['loss'] = dice_coef_loss

    if 'generator' in hyperparameters and 'rescale' in hyperparameters['generator']:
        hyperparameters['generator']['rescale'] = fix_rescale(hyperparameters['generator']['rescale'])
    if 'test_generator' in hyperparameters and 'rescale' in hyperparameters['test_generator']:
        hyperparameters['test_generator']['rescale'] = fix_rescale(hyperparameters['test_generator']['rescale'])
    return hyperparameters


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
    file_path = os.path.join(os.getcwd(),file_name)
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

# def get_hyperparameters(task):
#     task_dict = {
#         '1a': {'lr': .0001, 'batch_size': 8, 'epochs': 150, 'batch_norm': True, 'dropout': .5, 'optimizer': 'Adam',
#                 'loss': 'binary_crossentropy', 'metrics': ['dice_coef'], 'model': 'unet',
#                 'base': 16, 'input_shape': (256, 256, 1), 'data_path': '/Lab1/Lab3/X_ray/', 'test_size': .2,
#                 'last_layer_units': 1, 'last_layer_activation': 'sigmoid',
#                },
#         '1b': {'lr': .0001, 'batch_size': 8, 'epochs': 150, 'batch_norm': True, 'dropout': .5, 'optimizer': 'Adam',
#                'loss': 'dice_loss', 'metrics': ['dice_coef'], 'model': 'unet',
#                'base': 16, 'input_shape': (256, 256, 1), 'data_path': '/Lab1/Lab3/X_ray/', 'test_size': .2},
#         '2a': {'lr': .0001, 'batch_size': 8, 'epochs': 150, 'batch_norm': False, 'dropout': .5, 'optimizer': 'Adam',
#                'loss': 'binary_crossentropy', 'metrics': ['dice_coef'], 'model': 'unet',
#                'base': 16, 'input_shape': (256, 256, 1), 'data_path': '/Lab1/Lab3/X_ray/', 'test_size': .2},
#         '2b': {'lr': .0001, 'batch_size': 8, 'epochs': 150, 'batch_norm': False, 'dropout': .5, 'optimizer': 'Adam',
#                'loss': 'dice_loss', 'metrics': ['dice_coef'], 'model': 'unet',
#                'base': 16, 'input_shape': (256, 256, 1), 'data_path': '/Lab1/Lab3/X_ray/', 'test_size': .2},
#         '3': {'lr': .0001, 'batch_size': 8, 'epochs': 150, 'batch_norm': False, 'dropout': .5, 'optimizer': 'Adam',
#                'loss': 'dice_loss', 'metrics': ['dice_coef'], 'model': 'unet',
#                'base': 32, 'input_shape': (256, 256, 1), 'data_path': '/Lab1/Lab3/X_ray/', 'test_size': .2},
#         '4': {'lr': .0001, 'batch_size': 8, 'epochs': 150, 'batch_norm': False, 'dropout': .5, 'optimizer': 'Adam',
#                'loss': 'dice_loss', 'metrics': ['dice_coef'], 'model': 'unet',
#                'base': 16, 'input_shape': (256, 256, 1), 'data_path': '/Lab1/Lab3/X_ray/', 'test_size': .2,
#               'generator': {'rescale': 1. / 255, 'rotation_range': 10, 'width_shift_range': .1,
#                       'height_shift_range': .1, 'horizontal_flip': True, 'zoom_range': .2},
#               'test_generator': {'rescale': 1. / 255}},
#         '5a': {'lr': .0001, 'batch_size': 8, 'epochs': 150, 'batch_norm': True, 'dropout': .5, 'optimizer': 'Adam',
#                'loss': 'dice_loss', 'metrics': ['dice_coef'], 'model': 'unet',
#                'base': 16, 'input_shape': (256, 256, 1), 'data_path': '/Lab1/Lab3/CT/', 'test_size': .2},
#         '5a_BCE': {'lr': .0001, 'batch_size': 8, 'epochs': 150, 'batch_norm': True, 'dropout': .5, 'optimizer': 'Adam',
#                'loss': 'binary_crossentropy', 'metrics': ['dice_coef'], 'model': 'unet',
#                'base': 16, 'input_shape': (256, 256, 1), 'data_path': '/Lab1/Lab3/CT/', 'test_size': .2},
#         '5b': {'lr': .0001, 'batch_size': 8, 'epochs': 150, 'batch_norm': True, 'dropout': .5, 'optimizer': 'Adam',
#                'loss': 'binary_crossentropy', 'metrics': ['dice_coef', 'precision', 'recall'], 'model': 'unet',
#                'base': 16, 'input_shape': (256, 256, 1), 'data_path': '/Lab1/Lab3/CT/', 'test_size': .2,
#                'generator': {'rescale': 0, 'rotation_range': 10, 'width_shift_range': .1,
#                              'height_shift_range': .1, 'horizontal_flip': True, 'zoom_range': .2}},
#         '6': {'lr': .0001, 'batch_size': 8, 'epochs': 150, 'batch_norm': True, 'dropout': .5, 'optimizer': 'Adam',
#                'loss': 'binary_crossentropy', 'metrics': ['dice_coef', 'precision', 'recall'], 'model': 'unet',
#                'base': 16, 'input_shape': (256, 256, 1), 'data_path': '/Lab1/Lab3/CT/', 'test_size': .2,
#                'last_layer_units': 3, 'last_layer_activation':'softmax',
#                'generator': {'rescale': 0, 'rotation_range': 10, 'width_shift_range': .1,
#                              'height_shift_range': .1, 'horizontal_flip': True, 'zoom_range': .2}},
#         '7': {'lr': .0001, 'batch_size': 8, 'epochs': 150, 'batch_norm': True, 'dropout': .5, 'optimizer': 'Adam',
#                'loss': 'binary_crossentropy', 'metrics': ['dice_coef'], 'model': 'unet',
#                'base': 16, 'input_shape': (240, 240, 1), 'data_path': '/Lab1/Lab3/MRI/', 'test_size': .2,
#                'generator': {'rescale': 0, 'rotation_range': 10, 'width_shift_range': .1,
#                              'height_shift_range': .1, 'horizontal_flip': True, 'zoom_range': .2}},
#             }
#     return task_dict[task]
