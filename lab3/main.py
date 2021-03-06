import argparse
from utils import hyperparameters_processing, get_hyperparameters, plot_pair, update_board
from models import get_unet, plot_history
from data_loader import load_data
# import tensorflow as tf
# tf.config.gpu.set_per_process_memory_fraction(.6)
# tf.config.gpu.set_per_process_memory_growth(True)

parser = argparse.ArgumentParser()
parser.add_argument("-t", "--task", type=str, default="7",
                    help="Please enter tasks' numbers in a string separated by comma")

args = parser.parse_args()
tasks = args.task.split(',')
for task in tasks:
    hyperparameters = get_hyperparameters(task)
    hyperparameters = hyperparameters_processing(hyperparameters)
    train_data_gen, test_data_gen, total_train, total_val = load_data(hyperparameters)
    for i in range(10):
        batch_x, batch_y = next(train_data_gen)
        plot_pair(batch_x[0,:,:,0], batch_y[0,:,:,0])
        batch_x, batch_y = next(test_data_gen)
        plot_pair(batch_x[0,:,:,0], batch_y[0,:,:,0])
    model = get_unet(hyperparameters)
    model_history = model.fit_generator(train_data_gen,
                                        steps_per_epoch=total_train // hyperparameters['batch_size'],
                                        epochs=hyperparameters['epochs'],
                                        validation_data=test_data_gen,
                                        validation_steps=total_val // hyperparameters['batch_size'])
    plot_history(hyperparameters, model_history, task)
    update_board(hyperparameters, model.evaluate_generator(test_data_gen, steps=total_val), task)
