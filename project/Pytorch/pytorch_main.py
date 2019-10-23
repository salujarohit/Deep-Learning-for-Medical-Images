import argparse
from pytorch_utils import get_task_parameters, plot_pair, plot_triplet
from pytorch_models import get_unet, plot_history, save_model, save_step_prediction, plot_loss
from pytorch_dataloader import MyDataloader
from data_processing import PreProcessing
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
        self.model = self.get_model(self.hyperparameters).to(self.device).double()

        criterion = self.hyperparameters['loss']
        optimizer = self.hyperparameters['optimizer'](self.model.parameters(), lr=self.hyperparameters['lr'])
        for epoch in range(self.hyperparameters['epochs']):
            training_generator, validation_generator = self.data_loader.get_generators()
            running_loss = 0.0
            counter = 0.0
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
                running_loss += loss.item()

                loss.backward()
                optimizer.step()

                counter += 1
            losses.append(running_loss / counter)
            with torch.no_grad():
                val_running_loss = 0.0
                val_counter = 0.0
                for x_val, y_val in validation_generator:
                    x_val = x_val.to(self.device)
                    y_val = y_val.to(self.device)

                    self.model.eval()

                    yhat = self.model(x_val)
                    val_loss = criterion(y_val, yhat)
                    val_running_loss += val_loss.item()
                    val_counter += 1
            val_losses.append(val_running_loss / val_counter)
            print('epoch {}/{}, loss:{}, val_loss:{}'.format(epoch,self.hyperparameters['epochs'],running_loss / counter, val_running_loss / val_counter))
        print('Finished Training')
        plot_loss(self.current_task, losses, val_losses)

    def run_task(self):
        self.data_loader = self.get_dataloader(self.hyperparameters)
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
    parser.add_argument("-t", "--task", type=str, default="1b",
                        help="Please enter tasks' numbers in a string separated by comma")

    args = parser.parse_args()
    tasks = args.task.split(',')
    Simulation(tasks).simulate()


if __name__ == "__main__":
    main()