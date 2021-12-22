# ML4Science Project

## Repo organization

### Subdirectories

- data
	- Contains the all the data needed for the project
- analysis
	- Contains notebooks reffered to the analysis of the data
- training
	- Contains training notebooks and additional scripts

### Data
All of the data used for running scripts is stored in the ./data subdirectory which is not being maintained version controlled.
The data can be retrieved from [here](https://drive.google.com/drive/folders/1eKOZOaClqz94sYwslzmfj9ndhhTpvpo3?usp=sharing).
Note thas PESTO embedding data is kept on the private drive belonging to the EPFL's Laboratory for Biomolecular Modeling. For more informations, please contact us.

## Notebooks and scripts
- analysis
	- *data_analysis.ipynb* 
	- *data_analysis_LinearModel.ipynb* 
	- *PESTO_analysis.ipynb* 

- training
	- *datasets.py* - Contains implementations of the Dataset classes used in training for Linear Models (AASequenceDatasetLinear class) and CNN (AASequenceDataset).
	- *models.py* - Contains implementations of the model class used for the training of the CNN. Note that in case of the Linear model approach, models have been defined inside the notebook *trainingLinear.ipynb*
	- *training_and_evaluation* - Contains implementations of the trainig phase for Linear model and evaluation phase for both approaches
	- *utils.py* - Contains implementation of the util functions 
	- *trainingCNN* - Main notebook for the training and evaluation of the CNN model
	- *trainingLinear* - Main notebook for the trainig and evaluation of the Linear model