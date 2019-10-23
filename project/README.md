## This repository shows a segmentation pipeline developed for the project in the 'Deep Learning for Medical Images' course.
 
##### Pipeline is meant to help deep learning researchers in running different experiments easily by changing configureation file and running one command for the whole pipeline.   

##### We needed this pipeline to apply it on KiTS19 competition https://kits19.grand-challenge.org/home/ 

### Pipeline consists of three steps: preprocessing, main task, postprocessing.

Preprocessing is to read competition data and save it in smaller size with certain structure (i.e. /Image for images and /Mask for masks). 

Main task is where we train the model with availability of different kind of techniques (e.g. normal training, kfold training, weights maps, autocontext).

Postprocessing is meant for visualization and reporting, however for the project sake we use it for prediction.
 
 
We build two different pipelines; one using Keras and one using Pytorch.

For experimenting, please use Keras one. Pytorch pipeline is underdevelopment   
