import argparse
from keras_utils import get_task_parameters, plot_pair, plot_triplet
from keras_models import ModelContainer
from keras_dataloader import DataLoader, MyPredictionGenerator
import os
from data_processing import PreProcessing
import numpy as np
import nibabel as nib
from skimage.transform import resize
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
        self.data_loader = None
        self.model = None

    @staticmethod
    def get_dataloader(hyperparameters):
        return DataLoader(hyperparameters)

    @staticmethod
    def get_model_container(hyperparameters, current_task):
        return ModelContainer(hyperparameters, current_task)

    def preprocess_data(self):
        preprocessing_obj = PreProcessing(source=self.preprocess_parameters['source'],
                                          destination=self.preprocess_parameters['destination'], resize_shape=self.preprocess_parameters['resize_shape'])

        if self.preprocess_parameters.get('train'):
            preprocessing_obj.preprocess(self.preprocess_parameters['num_cases'],
                                     self.preprocess_parameters.get('starting_patient', 0))

        if self.preprocess_parameters.get('predict'):
            preprocessing_obj.preprocess_predictions(self.preprocess_parameters['num_cases_prediction'],
                                                     self.preprocess_parameters.get('starting_patient_prediction', 0))


    def run_task(self):
        self.data_loader = self.get_dataloader(self.hyperparameters)
        self.model_container = self.get_model_container(self.hyperparameters, self.current_task)
        self.model_container.train(self.data_loader)

    def postprocess_data(self):
        if self.postprocess_parameters.get('use_model'):
            self.model_container = self.get_model_container(self.postprocess_parameters, self.current_task)
        prediction_generator= MyPredictionGenerator(self.postprocess_parameters, len=self.postprocess_parameters['num_cases'])
        self.model_container.predict(prediction_generator, self.postprocess_parameters['start_pred'],
                                     self.postprocess_parameters['num_cases'], self.postprocess_parameters['data_path'])

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
    parser.add_argument("-t", "--task", type=str, default="test_example",
                        help="Please enter tasks' numbers in a string separated by comma")

    args = parser.parse_args()
    tasks = args.task.split(',')
    Simulation(tasks).simulate()


if __name__ == "__main__":
    main()