try:
    from tensorflow.keras.optimizers import Adam, SGD, RMSprop
except:
    from tensorflow.python.keras.optimizers import Adam, SGD, RMSprop

def hyperparameters_processing(hyperparameters):

    if hyperparameters['optimizer'] == "RMSprop":
        hyperparameters['optimizer'] = RMSprop
    elif hyperparameters['optimizer'] == "SGD":
        hyperparameters['optimizer'] = SGD
    else:
        hyperparameters['optimizer'] = Adam

    return hyperparameters

def get_hyperparameters(task):
    task_dict = {
        'test_local': {'lr': .0001, 'batch_size': 8, 'epochs': 5, 'batch_norm': False, 'dropout': [],
                 'spatial_dropout': [], 'optimizer': 'Adam', 'loss': 'binary_crossentropy',
                 'metrics': ['binary_accuracy'],
                 'model': 'vgg16', 'dense_units': [64, 64, 1], 'dense_activation': ['relu', 'relu', 'sigmoid'],
                 'base': 8, 'input_shape': (128, 128, 1), 'data_path': 'Data/Skin/', 'pattern': ['Mel', 'Nev'],
                 'use_gen': False},
        'test': {'lr': .0001, 'batch_size': 8, 'epochs': 5, 'batch_norm': False, 'dropout': [],
                'spatial_dropout': [], 'optimizer': 'Adam', 'loss': 'binary_crossentropy',
                'metrics': ['binary_accuracy'],
                'model': 'alexnet', 'dense_units': [64, 64, 1], 'dense_activation': ['relu', 'relu', 'sigmoid'],
                'base': 8, 'input_shape': (128, 128, 1), 'data_path': '/Lab1/Skin/', 'pattern': ['Mel', 'Nev'],
                'use_gen': False},
        'test1': {'lr': .0001, 'batch_size': 8, 'epochs': 2, 'batch_norm': False, 'dropout': [],
                 'spatial_dropout': [], 'optimizer': 'Adam', 'loss': 'binary_crossentropy',
                'metrics': ['binary_accuracy'],
                'model': 'alexnet', 'dense_units': [64, 64, 1], 'dense_activation': ['relu', 'relu', 'sigmoid'],
                 'base': 8, 'input_shape': (128, 128, 3), 'data_path': '/Lab1/Lab2/Skin/', 'pattern': ['Mel', 'Nev'],
                 'use_gen': True,
                 'gen':{'rescale':1./255,'rotation_range':45,'width_shift_range':.15,'height_shift_range':.15,'horizontal_flip':True,'zoom_range':0.5},
                 'gen_test':{'rescale':1./255,'rotation_range':0,'width_shift_range':0,'height_shift_range':0,'horizontal_flip':False,'zoom_range':0}},
        '1a': {'lr': .0001, 'batch_size': 8, 'epochs': 50, 'batch_norm': False, 'dropout': [],
                   'spatial_dropout': [], 'optimizer': 'Adam', 'loss': 'binary_crossentropy',
                'metrics': ['binary_accuracy'],
                'model': 'alexnet', 'dense_units': [64, 64, 1], 'dense_activation': ['relu', 'relu', 'sigmoid'],
                   'base': 8, 'input_shape': (128, 128, 1), 'data_path': '/Lab1/Skin/', 'pattern': ['Mel', 'Nev']},
        '1b': {'lr': .0001, 'batch_size': 8, 'epochs': 50, 'batch_norm': True, 'dropout': [],
                   'spatial_dropout': [], 'optimizer': 'Adam', 'loss': 'binary_crossentropy',
                'metrics': ['binary_accuracy'],
                'model': 'alexnet', 'dense_units': [64, 64, 1], 'dense_activation': ['relu', 'relu', 'sigmoid'],
                   'base': 8, 'input_shape': (128, 128, 1), 'data_path': '/Lab1/Skin/', 'pattern': ['Mel', 'Nev']},
        '1c1': {'lr': .00001, 'batch_size': 8, 'epochs': 80, 'batch_norm': False, 'dropout': [],
               'spatial_dropout': [], 'optimizer': 'Adam', 'loss': 'binary_crossentropy',
                'metrics': ['binary_accuracy'],
                'model': 'alexnet', 'dense_units': [64, 64, 1], 'dense_activation': ['relu', 'relu', 'sigmoid'],
               'base': 8, 'input_shape': (128, 128, 1), 'data_path': '/Lab1/Skin/', 'pattern': ['Mel', 'Nev'], 'use_gen': False},
        '1c2': {'lr': .00001, 'batch_size': 8, 'epochs': 80, 'batch_norm': True, 'dropout': [],
                'spatial_dropout': [], 'optimizer': 'Adam', 'loss': 'binary_crossentropy',
                'metrics': ['binary_accuracy'],
                'model': 'alexnet', 'dense_units': [64, 64, 1], 'dense_activation': ['relu', 'relu', 'sigmoid'],
                'base': 8, 'input_shape': (128, 128, 1), 'data_path': '/Lab1/Skin/', 'pattern': ['Mel', 'Nev'], 'use_gen': False},
        '2a': {'lr': .00001, 'batch_size': 8, 'epochs': 80, 'batch_norm': True, 'dropout': [.4, .4],
                'spatial_dropout': [], 'optimizer': 'Adam', 'loss': 'binary_crossentropy',
                'metrics': ['binary_accuracy'],
                'model': 'alexnet', 'dense_units': [64, 64, 1], 'dense_activation': ['relu', 'relu', 'sigmoid'],
                'base': 8, 'input_shape': (128, 128, 1), 'data_path': '/Lab1/Skin/', 'pattern': ['Mel', 'Nev'], 'use_gen': False},
        '2b': {'lr': .00001, 'batch_size': 8, 'epochs': 80, 'batch_norm': False, 'dropout': [.4, .4],
              'spatial_dropout': [], 'optimizer': 'Adam', 'loss': 'binary_crossentropy',
                'metrics': ['binary_accuracy'],
                'model': 'alexnet', 'dense_units': [64, 64, 1], 'dense_activation': ['relu', 'relu', 'sigmoid'],
              'base': 8, 'input_shape': (128, 128, 1), 'data_path': '/Lab1/Skin/', 'pattern': ['Mel', 'Nev'], 'use_gen': False},
        '3a': {'lr': .00001, 'batch_size': 8, 'epochs': 150, 'batch_norm': False, 'dropout': [.4, .4],
                'spatial_dropout': [.1, .1, .1], 'optimizer': 'Adam', 'loss': 'binary_crossentropy',
                'metrics': ['binary_accuracy'],
                'model': 'alexnet', 'dense_units': [64, 64, 1], 'dense_activation': ['relu', 'relu', 'sigmoid'],
                'base': 64, 'input_shape': (128, 128, 1), 'data_path': '/Lab1/Skin/', 'pattern': ['Mel', 'Nev'], 'use_gen': False},
        '3b': {'lr': .00001, 'batch_size': 8, 'epochs': 150, 'batch_norm': False, 'dropout': [],
               'spatial_dropout': [], 'optimizer': 'Adam', 'loss': 'binary_crossentropy',
                'metrics': ['binary_accuracy'],
                'model': 'alexnet', 'dense_units': [64, 64, 1], 'dense_activation': ['relu', 'relu', 'sigmoid'],
               'base': 64, 'input_shape': (128, 128, 1), 'data_path': '/Lab1/Skin/', 'pattern': ['Mel', 'Nev'], 'use_gen': False},
        '6': {'lr': .00001, 'batch_size': 8, 'epochs': 80, 'batch_norm': False, 'dropout': [.4, .4],
              'spatial_dropout': [], 'optimizer': 'Adam', 'loss': 'binary_crossentropy',
                'metrics': ['binary_accuracy'],
                'model': 'alexnet', 'dense_units': [128, 64, 1], 'dense_activation': ['relu', 'relu', 'sigmoid'],
              'base': 64, 'input_shape': (128, 128, 3), 'data_path': '/Lab1/Lab2/Skin/', 'pattern': ['Mel', 'Nev'],
              'use_gen': True,
              'gen': {'rescale': 1. / 255, 'rotation_range': 10, 'width_shift_range': .1,
                      'height_shift_range': .1, 'horizontal_flip': True, 'zoom_range': 0},
              'gen_test': {'rescale': 1. / 255, 'rotation_range': 0, 'width_shift_range': 0,
                           'height_shift_range': 0, 'horizontal_flip': False, 'zoom_range': 0}}
    }
    return task_dict[task]
