import argparse
from utils import hyperparameters_processing, get_hyperparameters, plot_pair, plot_triplet
from models import plot_history, save_model, get_model
from data_loader import get_data
import os
# import tensorflow as tf
# tf.config.gpu.set_per_process_memory_fraction(.6)
# tf.config.gpu.set_per_process_memory_growth(True)
from sklearn.preprocessing import MinMaxScaler
import keras

parser = argparse.ArgumentParser()
parser.add_argument("-t", "--task", type=str, default="3a",
                    help="Please enter tasks' numbers in a string separated by comma")

args = parser.parse_args()
tasks = args.task.split(',')
for task in tasks:
    hyperparameters = get_hyperparameters(task)
    hyperparameters = hyperparameters_processing(task, hyperparameters)
    model = get_model(hyperparameters)
    if 'use_generator' in hyperparameters and hyperparameters['use_generator']:
        train_data_generator, validation_data_generator = get_data(task, hyperparameters)
        model_history = model.fit_generator(train_data_generator, epochs=hyperparameters['epochs'],
                            validation_data=validation_data_generator)
    else:
        x_train, y_train, x_val, y_val = get_data(task, hyperparameters)
        model_history = model.fit(x_train, y_train, batch_size=hyperparameters['batch_size'], validation_data=(x_val, y_val), epochs=hyperparameters['epochs'])

    plot_history(hyperparameters, model_history, task)
    if 'save_model' in hyperparameters and hyperparameters['save_model']:
        save_model(model, task)

    # predicted_stock_price = model.predict(x_val)
    # predicted_stock_price = sc.inverse_transform(predicted_stock_price)
    # gt = sc.inverse_transform(y_val)
    # metric = keras.losses.mean_absolute_error(gt, predicted_stock_price)
    # print(metric)