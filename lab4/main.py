import argparse
from utils import hyperparameters_processing, get_hyperparameters, plot_pair, plot_triplet
from models import get_unet, plot_history, save_model, save_step_prediction
from data_loader import get_folds, get_data_with_generator_on_the_fly
import os
# import tensorflow as tf
# tf.config.gpu.set_per_process_memory_fraction(.6)
# tf.config.gpu.set_per_process_memory_growth(True)

parser = argparse.ArgumentParser()
parser.add_argument("-t", "--task", type=str, default="3a",
                    help="Please enter tasks' numbers in a string separated by comma")

args = parser.parse_args()
tasks = args.task.split(',')
for task in tasks:
    hyperparameters = get_hyperparameters(task)
    hyperparameters = hyperparameters_processing(hyperparameters)
    folds = get_folds(hyperparameters)
    autocontext_step = hyperparameters['autocontext_step']
    model_predictions = [None] * len(os.listdir(os.path.join(os.getcwd(), os.path.join(hyperparameters['data_path'], 'Image'))))
    for s_step in range(0, autocontext_step):
        folds = get_folds(hyperparameters)
        for fold_num, fold in enumerate(folds):
            train_images, train_masks, validation_images, validation_masks = fold
            print("train images", train_images)
            train_data_gen, test_data_gen = get_data_with_generator_on_the_fly(hyperparameters, train_images,
                                                                               train_masks, validation_images,
                                                                               validation_masks, s_step, fold_num,
                                                                               len(validation_images))
            model = get_unet(hyperparameters)
            # for i in range(10):
            #     batch_x_w, batch_y = train_data_gen.__getitem__(i)
            #     plot_triplet(batch_x_w[0][0,:,:,0], batch_x_w[1][0,:,:,0], batch_y[0,:,:,0])
            #     batch_x_w, batch_y = test_data_gen.__getitem__(i)
            #     plot_triplet(batch_x_w[0][0,:,:,0], batch_x_w[1][0,:,:,0], batch_y[0,:,:,0])
            model_history = model.fit_generator(train_data_gen,
                                                # steps_per_epoch=total_train // hyperparameters['batch_size'],
                                                epochs=hyperparameters['epochs'],
                                                validation_data=test_data_gen)
            plot_history(hyperparameters, model_history, task, s_step, fold_num)
            if 'save_model' in hyperparameters and hyperparameters['save_model']:
                save_model(model, task, s_step, fold_num)

            y_pred = model.predict(test_data_gen)
            total_val = len(test_data_gen.image_filenames)
            model_predictions[(fold_num * total_val):((fold_num + 1) * total_val)] = y_pred
        save_step_prediction(model_predictions, s_step)
