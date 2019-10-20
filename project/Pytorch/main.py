import argparse
from Pytorch.utils import get_task_parameters, plot_pair, plot_triplet
from Pytorch.models import get_unet, plot_history, save_model, save_step_prediction, plot_loss
from Pytorch.dataloader import MyDataloader
import os
from Pytorch.data_processing import PreProcessing
import torch

class Simulation:
    def __init__(self, tasks):
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.tasks = tasks
        self.current_task = None
        self.preprocess_parameters = {}
        self.hyperparameters = {}
        self.postprocess_parameters = {}

    @staticmethod
    def get_dataloader(hyperparameters):
        return MyDataloader(hyperparameters)

    @staticmethod
    def get_model(hyperparameters):
        return get_unet(hyperparameters)

    def normal_train(self):
        losses = []
        val_losses = []
        self.model = self.get_model(self.hyperparameters).to(self.device)
        criterion = self.hyperparameters['loss']
        optimizer = self.hyperparameters['optimizer'](self.model.parameters(), lr=self.hyperparameters['lr'])#, momentum=0.9)
        for epoch in range(self.hyperparameters['epochs']):  # loop over the dataset multiple times
            training_generator, validation_generator = self.data_loader.get_generators()
            running_loss = 0.0
            val_running_loss = 0.0
            for i, data in enumerate(training_generator, 0):
                # get the inputs; data is a list of [inputs, labels]
                if self.hyperparameters.get('use_weight_maps'):
                    inputs, labels, mask_boundary = data
                else:
                    inputs, labels = data
                inputs = inputs.to(self.device)
                labels = labels.to(self.device)
                self.model.train()
                # zero the parameter gradients
                optimizer.zero_grad()

                # forward + backward + optimize
                outputs = self.model(inputs)
                loss = criterion(outputs, labels)
                losses.append(loss.item())
                loss.backward()
                optimizer.step()

                with torch.no_grad():
                    for x_val, y_val in validation_generator:
                        x_val = x_val.to(self.device)
                        y_val = y_val.to(self.device)

                        self.model.eval()

                        yhat = self.model(x_val)
                        val_loss = criterion(y_val, yhat)
                        val_losses.append(val_loss.item())

                # print statistics
                running_loss += loss.item()
                val_running_loss += val_loss.item()
                if i % 2000 == 1999:  # print every 2000 mini-batches
                    print('[%d, %5d] loss: %.3f val_loss: %.3f' %
                          (epoch + 1, i + 1, running_loss / 2000, val_running_loss / 2000))
                    running_loss = 0.0
                    val_running_loss = 0.0

        print('Finished Training')

        # for batch_x, batch_y in training_generator:
        #     plot_pair(batch_x[0,:,:,0], batch_y[0,:,:,0])
        #     plot_pair(batch_x[0, :, :, 0], batch_y[0, :, :, 1])
        #     plot_pair(batch_x[0, :, :, 0], batch_y[0, :, :, 2])
        plot_loss(self.current_task, losses, val_loss)
        # if self.hyperparameters.get('save_model'):
        #     save_model(self.model, self.current_task)

    # def kfold_train(self, step_num=None):
    #     for fold_num in range(self.hyperparameters['folds']):
    #         training_generator, validation_generator = self.data_loader.get_generators(fold_num=fold_num, step_num=step_num)
    #         self.model = self.get_model(self.hyperparameters)
    #         model_history = self.model.fit_generator(training_generator,
    #                                                  epochs=self.hyperparameters['epochs'],
    #                                                  validation_data=validation_generator)
    #         plot_history(self.hyperparameters, model_history, self.current_task, step_num=step_num, fold_num=fold_num)
    #         if self.hyperparameters.get('save_model'):
    #             save_model(self.model, self.current_task, step_num=step_num, fold_num=fold_num)

    # def autcontext_train(self):
    #     autocontext_step = self.hyperparameters['autocontext_step']
    #     model_predictions = [None] * len(
    #         os.listdir(os.path.join(os.getcwd(), os.path.join(self.hyperparameters['data_path'], 'Image'))))
    #     for step_num in range(0, autocontext_step):
    #         for fold_num in range(self.hyperparameters['folds']):
    #             training_generator, validation_generator = self.data_loader.get_generators(fold_num, step_num)
    #             self.model = self.get_model(self.hyperparameters)
    #             model_history = self.model.fit_generator(training_generator,
    #                                                 epochs=self.hyperparameters['epochs'],
    #                                                 validation_data=validation_generator)
    #             plot_history(self.hyperparameters, model_history, self.current_task, step_num, fold_num)
    #             if self.hyperparameters.get('save_model'):
    #                 save_model(self.model, self.current_task, step_num, fold_num)
    #
    #             y_pred = self.model.predict(validation_generator)
    #             total_val = len(validation_generator.image_filenames)
    #             model_predictions[(fold_num * total_val):((fold_num + 1) * total_val)] = y_pred
    #         save_step_prediction(model_predictions, step_num)

    def run_task(self):
        self.data_loader = self.get_dataloader(self.hyperparameters)
        # if self.hyperparameters.get('autocontext_step'):
        #     self.autcontext_train()
        # elif self.hyperparameters.get('folds'):
        #     self.kfold_train()
        # else:
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
    parser.add_argument("-t", "--task", type=str, default="1a",
                        help="Please enter tasks' numbers in a string separated by comma")

    args = parser.parse_args()
    tasks = args.task.split(',')
    Simulation(tasks).simulate()


if __name__ == "__main__":
    main()