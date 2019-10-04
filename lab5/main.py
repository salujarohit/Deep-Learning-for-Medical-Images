import argparse
from utils import hyperparameters_processing, get_hyperparameters, plot_pair, plot_triplet
from models import get_unet, plot_history, save_model, save_step_prediction, get_model
from data_loader import get_folds, get_data_with_generator_on_the_fly, get_data, MyBatchGenerator
import os
# import tensorflow as tf
# tf.config.gpu.set_per_process_memory_fraction(.6)
# tf.config.gpu.set_per_process_memory_growth(True)

parser = argparse.ArgumentParser()
parser.add_argument("-t", "--task", type=str, default="1a",
                    help="Please enter tasks' numbers in a string separated by comma")

args = parser.parse_args()
tasks = args.task.split(',')
for task in tasks:
    hyperparameters = get_hyperparameters(task)
    hyperparameters = hyperparameters_processing(task, hyperparameters)
    x_train, y_train, x_val, y_val = get_data(task, hyperparameters)
    model = get_model(hyperparameters)
    if 'use_generator' in hyperparameters and hyperparameters['use_generator']:
        model_history = model.fit_generator(MyBatchGenerator(x_train, y_train, batch_size=1), epochs=hyperparameters['epochs'],
                            validation_data=MyBatchGenerator(x_val, y_val, batch_size=1), validation_steps=len(x_val))
    else:
        model_history = model.fit(x_train, y_train, batch_size=hyperparameters['batch_size'], validation_data=(x_val, y_val), epochs=hyperparameters['epochs'])
    plot_history(hyperparameters, model_history, task)
    if 'save_model' in hyperparameters and hyperparameters['save_model']:
        save_model(model, task)

