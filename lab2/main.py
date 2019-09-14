import argparse
from utils import hyperparameters_processing, get_hyperparameters
from models import get_model, plot_history
from data_loader import load_data
import tensorflow as tf
tf.config.gpu.set_per_process_memory_fraction(.3)
tf.config.gpu.set_per_process_memory_growth(True)

parser = argparse.ArgumentParser()
parser.add_argument("-t", "--task", type=str, default="test", help="Please enter tasks' numbers in a string separated by comma")

data_path = ''
use_gen = False
args = parser.parse_args()
tasks = args.task.split(',')
for task in tasks:
    hyperparameters= get_hyperparameters(task)
    hyperparameters = hyperparameters_processing(hyperparameters)
    if data_path == '' or hyperparameters['data_path'] != data_path or hyperparameters['use_gen'] != use_gen:
        if hyperparameters['use_gen']:
           train_data_gen, test_data_gen, total_train, total_val = load_data(hyperparameters)
        else:
            x_train, x_test, y_train, y_test = load_data(hyperparameters)

        data_path = hyperparameters['data_path']
        use_gen = hyperparameters['use_gen']

    model = get_model(hyperparameters)
    if hyperparameters['use_gen']:
        history = model.fit_generator(train_data_gen,
            steps_per_epoch=total_train // hyperparameters['batch_size'], epochs=hyperparameters['epochs'],
            validation_data=test_data_gen, validation_steps=total_val // hyperparameters['batch_size'])
        print(model.evaluate_generator(test_data_gen))
    else:
        model_history = model.fit(x_train, y_train, validation_data=(x_test, y_test), epochs=hyperparameters['epochs'], batch_size=hyperparameters['batch_size'], verbose=0)
        print(model.evaluate(x_test, y_test))

    plot_history(model_history, task)
