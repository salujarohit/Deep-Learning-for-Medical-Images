import argparse
from utils import hyperparameters_processing, get_hyperparameters, plot_pair, plot_triplet
from models import get_unet, plot_history, save_model
from data_loader import get_folds, get_data_with_generator_on_the_fly
import os
# import tensorflow as tf
# tf.config.gpu.set_per_process_memory_fraction(.6)
# tf.config.gpu.set_per_process_memory_growth(True)

parser = argparse.ArgumentParser()
parser.add_argument("-t", "--task", type=str, default="2a",
                    help="Please enter tasks' numbers in a string separated by comma")

args = parser.parse_args()
tasks = args.task.split(',')
for task in tasks:
    hyperparameters = get_hyperparameters(task)
    hyperparameters = hyperparameters_processing(hyperparameters)
    folds = get_folds(hyperparameters)
    for i, fold in enumerate(folds):
        train_images, train_masks, validation_images, validation_masks = fold
        train_data_gen, test_data_gen, total_train, total_val = get_data_with_generator_on_the_fly(hyperparameters,
                                                                                                   train_images,
                                                                                                   train_masks,
                                                                                                   validation_images,
                                                                                                   validation_masks)
        model = get_unet(hyperparameters)
        # for i in range(10):
        #     batch_x_w, batch_y = train_data_gen.__getitem__(i)
        #     plot_triplet(batch_x_w[0][0,:,:,0], batch_x_w[1][0,:,:,0], batch_y[0,:,:,0])
        #     batch_x_w, batch_y = test_data_gen.__getitem__(i)
        #     plot_triplet(batch_x_w[0][0,:,:,0], batch_x_w[1][0,:,:,0], batch_y[0,:,:,0])
        model_history = model.fit_generator(train_data_gen,
                                            steps_per_epoch=total_train // hyperparameters['batch_size'],
                                            epochs=hyperparameters['epochs'],
                                            validation_data=test_data_gen,
                                            validation_steps=total_val // hyperparameters['batch_size'])
        plot_history(hyperparameters, model_history, task, i)
        if 'save_model' in hyperparameters and hyperparameters['save_model']:
            save_model(model, task, i)
        print(i)
    # update_board(hyperparameters, model.evaluate_generator(test_data_gen, steps=total_val), task)
