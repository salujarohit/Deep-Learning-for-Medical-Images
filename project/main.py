import argparse
from utils import hyperparameters_processing, get_hyperparameters, plot_pair, plot_triplet
from models import get_unet, plot_history, save_model, save_step_prediction
from dataloader import DataLoader
import os
# import tensorflow as tf
# tf.config.gpu.set_per_process_memory_fraction(.6)
# tf.config.gpu.set_per_process_memory_growth(True)

class Simulation:
    def __init__(self, tasks):
        self.tasks = tasks

    @staticmethod
    def get_dataloader(hyperparameters):
        return DataLoader(hyperparameters)

    @staticmethod
    def get_model(hyperparameters):
        return get_unet(hyperparameters)

    def normal_train(self, task):
        self.model = self.get_model(self.hyperparameters)
        training_generator, validation_generator = self.data_loader.get_generators()
        model_history = self.model.fit_generator(training_generator,
                                                 epochs=self.hyperparameters['epochs'],
                                                 validation_data=validation_generator)
        plot_history(self.hyperparameters, model_history, task)
        if self.hyperparameters.get('save_model'):
            save_model(self.model, task)

    def kfold_train(self, task, step_num=None):
        for fold_num in range(self.hyperparameters['folds']):
            training_generator, validation_generator = self.data_loader.get_generators(fold_num=fold_num, step_num=step_num)
            self.model = self.get_model(self.hyperparameters)
            model_history = self.model.fit_generator(training_generator,
                                                     epochs=self.hyperparameters['epochs'],
                                                     validation_data=validation_generator)
            plot_history(self.hyperparameters, model_history, task, step_num=step_num, fold_num=fold_num)
            if self.hyperparameters.get('save_model'):
                save_model(self.model, task, step_num=step_num, fold_num=fold_num)

    def autcontext_train(self, task):
        autocontext_step = self.hyperparameters['autocontext_step']
        model_predictions = [None] * len(
            os.listdir(os.path.join(os.getcwd(), os.path.join(self.hyperparameters['data_path'], 'Image'))))
        for step_num in range(0, autocontext_step):
            for fold_num in range(self.hyperparameters['folds']):
                training_generator, validation_generator = self.data_loader.get_generators(fold_num, step_num)
                self.model = self.get_model(self.hyperparameters)
                model_history = self.model.fit_generator(training_generator,
                                                    epochs=self.hyperparameters['epochs'],
                                                    validation_data=validation_generator)
                plot_history(self.hyperparameters, model_history, task, step_num, fold_num)
                if self.hyperparameters.get('save_model'):
                    save_model(self.model, task, step_num, fold_num)

                y_pred = self.model.predict(validation_generator)
                total_val = len(validation_generator.image_filenames)
                model_predictions[(fold_num * total_val):((fold_num + 1) * total_val)] = y_pred
            save_step_prediction(model_predictions, step_num)

    def run_task(self, task):
        self.hyperparameters = get_hyperparameters(task)
        self.data_loader = self.get_dataloader(self.hyperparameters)
        if self.hyperparameters.get('autocontext_step'):
            self.autcontext_train(task)
        elif self.hyperparameters.get('folds'):
            self.kfold_train(task)
        else:
            self.normal_train(task)

    def simulate(self):
        for task in self.tasks:
            self.run_task(task)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("-t", "--task", type=str, default="1a",
                        help="Please enter tasks' numbers in a string separated by comma")

    args = parser.parse_args()
    tasks = args.task.split(',')
    Simulation(tasks).simulate()

if __name__ == "__main__":
    main()