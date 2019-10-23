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
    - "model": String, model to be used for training (e.g. "unet")
    - "base": 16,
    - "input_shape": [240, 240, 1],
    - "data_path": "/dl_data/",
    - "test_size": 0.2,
    - "last_layer_units": 3,
    - "last_layer_activation": "softmax",
    - "save_model": true,
    - "shuffle": true