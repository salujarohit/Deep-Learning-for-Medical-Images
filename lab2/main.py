import argparse
from utils import hyperparameters_processing, get_hyperparameters
from models import get_model, plot_history
from data_loader import get_data
parser = argparse.ArgumentParser()
parser.add_argument("-t", "--task", type=str, default="1a", help="Please enter task number")
# parser.add_argument("-m", "--model", type=str, default="alexnet", help="Please enter task number")
# parser.add_argument("-lr", "--learning_rate", type=float, default=.00001, help="Please enter task number")
# parser.add_argument("-bs", "--batch_size", type=int, default="8", help="Please enter task number")
# parser.add_argument("-e", "--epoch", type=int, default=20, help="Please enter task number")
# parser.add_argument("-t", "--task", type=str, default="1a", help="Please enter task number")
# parser.add_argument("-t", "--task", type=str, default="1a", help="Please enter task number")
# parser.add_argument("-t", "--task", type=str, default="1a", help="Please enter task number")
# parser.add_argument("-t", "--task", type=str, default="1a", help="Please enter task number")

args = parser.parse_args()
hyperparameters= get_hyperparameters(args.task)
hyperparameters = hyperparameters_processing(hyperparameters)
x_train, x_test, y_train, y_test = get_data(hyperparameters['data_path'], hyperparameters['input_shape'][0],
                                             hyperparameters['input_shape'][0], hyperparameters['pattern'])
print(x_train.shape,x_test.shape,y_train.shape,y_test.shape)
model = get_model(hyperparameters)
model_history = model.fit(x_train, y_train, validation_data=(x_test, y_test), epochs=hyperparameters['epochs'], batch_size=hyperparameters['batch_size'], verbose=0)
plot_history(model_history)
print(model.evaluate(x_test, y_test))
