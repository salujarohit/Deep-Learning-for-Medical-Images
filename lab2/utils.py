try:
    from tensorflow.keras.optimizers import Adam, SGD, RMSprop
except:
    from tensorflow_core.python.keras.optimizers import Adam, SGD, RMSprop

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
        '1a': {'lr': .0001, 'batch_size': 8, 'epochs': 50, 'batch_norm': False, 'dropout': [],
                   'spatial_dropout': [], 'optimizer': 'Adam', 'loss': 'binary_crossentropy', 'model': 'alexnet',
                   'base': 8, 'input_shape': (128, 128, 1), 'data_path': 'Data/Skin/', 'pattern': ['Mel', 'Nev']},
        '1b': {'lr': .0001, 'batch_size': 8, 'epochs': 50, 'batch_norm': True, 'dropout': [],
                   'spatial_dropout': [], 'optimizer': 'Adam', 'loss': 'binary_crossentropy', 'model': 'alexnet',
                   'base': 8, 'input_shape': (128, 128, 1), 'data_path': 'Data/Skin/', 'pattern': ['Mel', 'Nev']},
        '1c1': {'lr': .00001, 'batch_size': 8, 'epochs': 80, 'batch_norm': False, 'dropout': [],
               'spatial_dropout': [], 'optimizer': 'Adam', 'loss': 'binary_crossentropy', 'model': 'alexnet',
               'base': 8, 'input_shape': (128, 128, 1), 'data_path': 'Data/Skin/', 'pattern': ['Mel', 'Nev']},
        '1c2': {'lr': .00001, 'batch_size': 8, 'epochs': 80, 'batch_norm': True, 'dropout': [],
                'spatial_dropout': [], 'optimizer': 'Adam', 'loss': 'binary_crossentropy', 'model': 'alexnet',
                'base': 8, 'input_shape': (128, 128, 1), 'data_path': 'Data/Skin/', 'pattern': ['Mel', 'Nev']},
        '2a': {'lr': .00001, 'batch_size': 8, 'epochs': 80, 'batch_norm': True, 'dropout': [.4, .4],
                'spatial_dropout': [], 'optimizer': 'Adam', 'loss': 'binary_crossentropy', 'model': 'alexnet',
                'base': 8, 'input_shape': (128, 128, 1), 'data_path': 'Data/Skin/', 'pattern': ['Mel', 'Nev']},
        '2b': {'lr': .00001, 'batch_size': 8, 'epochs': 80, 'batch_norm': False, 'dropout': [.4, .4],
              'spatial_dropout': [], 'optimizer': 'Adam', 'loss': 'binary_crossentropy', 'model': 'alexnet',
              'base': 8, 'input_shape': (128, 128, 1), 'data_path': 'Data/Skin/', 'pattern': ['Mel', 'Nev']},
        '3a': {'lr': .00001, 'batch_size': 8, 'epochs': 150, 'batch_norm': False, 'dropout': [.4, .4],
                'spatial_dropout': [.1, .1, .1], 'optimizer': 'Adam', 'loss': 'binary_crossentropy', 'model': 'alexnet',
                'base': 64, 'input_shape': (128, 128, 1), 'data_path': 'Data/Skin/', 'pattern': ['Mel', 'Nev']},
        '3b': {'lr': .00001, 'batch_size': 8, 'epochs': 150, 'batch_norm': False, 'dropout': [],
               'spatial_dropout': [], 'optimizer': 'Adam', 'loss': 'binary_crossentropy', 'model': 'alexnet',
               'base': 64, 'input_shape': (128, 128, 1), 'data_path': 'Data/Skin/', 'pattern': ['Mel', 'Nev']}
    }
    return task_dict[task]