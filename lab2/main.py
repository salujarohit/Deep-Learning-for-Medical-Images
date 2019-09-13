import argparse
from utils import hyperparameters_processing, get_hyperparameters
from models import get_model, plot_history
from data_loader import get_data
parser = argparse.ArgumentParser()
parser.add_argument("-t", "--task", type=str, default="test", help="Please enter tasks' numbers in a string separated by comma")

data_path = ''
args = parser.parse_args()
tasks = args.task.split(',')
for task in tasks:
    hyperparameters= get_hyperparameters(task)
    hyperparameters = hyperparameters_processing(hyperparameters)
    if data_path == '' or hyperparameters['data_path'] != data_path:
        x_train, x_test, y_train, y_test = get_data(hyperparameters['data_path'], hyperparameters['input_shape'][0],
                                                     hyperparameters['input_shape'][0], hyperparameters['pattern'])
        data_path = hyperparameters['data_path']

    model = get_model(hyperparameters)
    model_history = model.fit(x_train, y_train, validation_data=(x_test, y_test), epochs=hyperparameters['epochs'], batch_size=hyperparameters['batch_size'], verbose=0)
    plot_history(model_history, task)
    print(model.evaluate(x_test, y_test))
