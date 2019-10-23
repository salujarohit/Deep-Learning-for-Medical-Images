### For experiments: 
Create a json configuration file inside "tasks" folder.
Inside this file you can have one or more of next three elements:
 
- preprocess_data: which is responsible for reading .nii.gz data files and separate them into /data_path/Image & /data_path/Mask
    - "source": string, data path for source data files (e.g. "/dl_data/raw/")
    - "destination": string, data path for output directories(/Image, /Mask) (e.g. "/dl_data/")
    - "resize_shape": list, showing required shape for resizing (e.g. [240,240])
    - "train": boolean, showing if extraction needed for training or not (e.g. true)
    - "num_cases": Integer, showing number of cases(patients) to extract their data for training (e.g. 210)
    - "starting_patient": Integer, showing which patient to start extraction with and then move forward (e.g. 0)
    - "predict": boolean, showing if extraction needed for prediction or not (e.g. true)
    - "num_cases_prediction": Integer, showing number of cases(patients) to extract their data for prediction (e.g. 90)
    - "starting_patient_prediction": Integer, showing which patient to start extraction with for prediction and then move forward (e.g. 210)   

- train_parameters: which is responsible for training the model using following hyperparameters: 
    - "lr": float, learning rate used for the optimizer (e.g.0.0001)
    - "batch_size": Integer, batch size used for training (e.g. 8)
    - "epochs": Integer, number of training epochs (e.g.2)
    - "batch_norm": boolean, whether to use batch normalization inside the model or not (e.g.true)
    - "dropout": float, dropout percentage (e.g.0.5)
    - "optimizer": String, which optimizer used for training (e.g. "Adam")
    - "loss": String, which loss function should be used during the training (e.g. "competition_loss")
    - "metrics": list of strings, metrics used to be shown after training (e.g.["competition_coef", "custom_competition_coef", "dice_coef", "precision", "recall"])
    - "model": String, model to be used for training (e.g. "unet"), shouldn't be used if "use_model" is used
    - "use_model": String, for the model to be used to continue training (e.g. "1d.h5"), shouldn't be used if "model" is used
    - "base": Integer, number of features used for the first layer in U-net and then multiplies by 2 for next layers (e.g.16)
    - "input_shape": list, showing required shape for model input (e.g. [240, 240, 1])
    - "data_path": string, data path for source data files (e.g. "/dl_data/")
    - "test_size": float, showing percentage of size of data used for validation (e.g. 0.2), shouldn't be used when "folds" is used.
    - "folsds": Integer, showing how many folds to be used when applying kfold technique, shouldn't be used when "test_size" is used. 
    - "use_weight_maps": bool, whether to use weight maps or not, default false.
    - "autocontext_step": Integer, showing how many steps to be used for autocontext. If it is mentioned, autocontext will be used. 
    - "last_layer_units": Integer, number of units used for last layer (number of classes when softmax used as activation or 1 when when sigmoid used as activation)
    - "last_layer_activation": String, used for last layer activation (e.g. "softmax")
    - "save_model": boolean, whether to save the model or not
    - "shuffle": boolean, whether to shuffle data before training or not.
    
- postprocess_data
    - "loss": String, which loss function used during the training (e.g. "competition_loss"), don't have to be mentioned if prediction happening right after training, and should be used when you use a saved model for prediction. 
    - "metrics": list of strings, metrics used in the training (e.g.["competition_coef", "custom_competition_coef", "dice_coef", "precision", "recall"]), don't have to be mentioned if prediction happening right after training, and should be used when you use a saved model for prediction.
    - "use_model": String , which model to use for prediction (e.g. "1j.h5")
    - "start_pred": Integer, showing which patient to start prediction with (e.g. 210)
    - "num_cases": Integer, showing number of cases(patients) to predict (e.g. 90)
    - "data_path": string, data path for source data files for prediction (e.g. "/dl_data/")
    - "input_shape": list, showing required shape for model input (e.g. [240, 240, 1])
    
#### For an example, please check tasks/test_example.json
