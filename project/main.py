import argparse
from utils import get_task_parameters, plot_pair, plot_triplet
from models import get_unet, plot_history, save_model, save_step_prediction
from dataloader import DataLoader
import os
from data_processing import PreProcessing
import tensorflow as tf
# tf.config.gpu.set_per_process_memory_fraction(.6)
# tf.config.gpu.set_per_process_memory_growth(True)
class Simulation:
    def __init__(self, tasks):
        self.tasks = tasks
        self.current_task = None
        self.preprocess_parameters = {}
        self.hyperparameters = {}
        self.postprocess_parameters = {}

    @staticmethod
    def get_dataloader(hyperparameters):
        return DataLoader(hyperparameters)

    @staticmethod
    def get_model(hyperparameters):
        return get_unet(hyperparameters)

    def normal_train(self):
        self.model = self.get_model(self.hyperparameters)
        training_generator, validation_generator = self.data_loader.get_generators()
        # for batch_x, batch_y in training_generator:
        #     plot_pair(batch_x[0,:,:,0], batch_y[0,:,:,0])
        #     plot_pair(batch_x[0, :, :, 0], batch_y[0, :, :, 1])
        #     plot_pair(batch_x[0, :, :, 0], batch_y[0, :, :, 2])
        model_history = self.model.fit_generator(training_generator,
                                                 epochs=self.hyperparameters['epochs'],
                                                 validation_data=validation_generator)
        plot_history(self.hyperparameters, model_history, self.current_task)
        if self.hyperparameters.get('save_model'):
            save_model(self.model, self.current_task)

    def kfold_train(self, step_num=None):
        for fold_num in range(self.hyperparameters['folds']):
            training_generator, validation_generator = self.data_loader.get_generators(fold_num=fold_num, step_num=step_num)
            self.model = self.get_model(self.hyperparameters)
            model_history = self.model.fit_generator(training_generator,
                                                     epochs=self.hyperparameters['epochs'],
                                                     validation_data=validation_generator)
            plot_history(self.hyperparameters, model_history, self.current_task, step_num=step_num, fold_num=fold_num)
            if self.hyperparameters.get('save_model'):
                save_model(self.model, self.current_task, step_num=step_num, fold_num=fold_num)

    def autcontext_train(self):
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
                plot_history(self.hyperparameters, model_history, self.current_task, step_num, fold_num)
                if self.hyperparameters.get('save_model'):
                    save_model(self.model, self.current_task, step_num, fold_num)

                y_pred = self.model.predict(validation_generator)
                total_val = len(validation_generator.image_filenames)
                model_predictions[(fold_num * total_val):((fold_num + 1) * total_val)] = y_pred
            save_step_prediction(model_predictions, step_num)

    def run_task(self):
        self.data_loader = self.get_dataloader(self.hyperparameters)
        if self.hyperparameters.get('autocontext_step'):
            self.autcontext_train()
        elif self.hyperparameters.get('folds'):
            self.kfold_train()
        else:
            self.normal_train()

    def preprocess_data(self):
        preprocessing_obj = PreProcessing(source=self.preprocess_parameters['source'],
                                          destination=self.preprocess_parameters['destination'])
        preprocessing_obj.preprocess(self.preprocess_parameters['num_cases'],self.preprocess_parameters.get('starting_patient', 0))

    def postprocess_data(self):
        pass

    def separate_parameters(self, task_parameters):
        if task_parameters.get('train_parameters'):
            self.hyperparameters = task_parameters['train_parameters']
        if task_parameters.get('preprocess_data'):
            self.preprocess_parameters = task_parameters['preprocess_data']
        if task_parameters.get('postprocess_data'):
            self.postprocess_parameters = task_parameters['postprocess_data']

    def simulate(self):
        for task in self.tasks:
            self.current_task = task
            task_parameters = get_task_parameters(task)
            self.separate_parameters(task_parameters)
            if self.preprocess_parameters:
                self.preprocess_data()
            if self.hyperparameters:
                self.run_task()
            if self.postprocess_parameters:
                self.postprocess_data()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("-t", "--task", type=str, default="0",
                        help="Please enter tasks' numbers in a string separated by comma")

    args = parser.parse_args()
    tasks = args.task.split(',')
    Simulation(tasks).simulate()


if __name__ == "__main__":
    main()