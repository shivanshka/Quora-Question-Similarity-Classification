# Quora Question Similarity Detection
Built a system which detect semantically and contextually similar Question in Quora. It will be helpfull for platforms like Quora to make their services more optimized and efficient.

## Tools Used
- Jupyter Notebook
- VS Code
- Flask
- Machine Learning Algorithms: Random Forest Classifeir and XGBoost Classifier
- MLOps
- HTML

## Dataset
We have taken data from Kaggle. It was a competition data with around 4 lacs datapoints in training dataset and 30,000 datapoints in Test dataset.

data link: https://www.kaggle.com/competitions/quora-question-pairs/data

## Project Details
There are six packages in the pipeline: Config, Entity, Constant, Exception, Logger, Components and Pipeline

### Config
This package will create all folder structures and provide inputs to the each of the components.

### Entity
This package will defines named tuple for each of the components config and artifacts it generates.

### Constant
This package will contain all predefined constants which can be used accessed from anywhere

### Exception
This package contains the custom exception class for the Prediction Appliaction

### Logger
This package helps in logging all the activity

### Components
This package contains five modules:
1. Data Ingestion: This module downloads the data from the link, unzip it, then stores entire data into Db.
                   From DB it extracts all data into single csv file and split it into training and testing datasets.
2. Data Validation: This module validates whether data files passed are as per defined schema which was agreed upon
                    by client.
3. Data Transformation: This module applies all the Feature Engineering and preprocessing to the data we need to 
                        train our model and save  the pickle object for same.
4. Model Trainer: This module trains the model on transformed data, evalutes it based on R2 accuracy score and 
                  saves the best performing model object for prediction

### Pipeline
This package contains two modules:
1. Training Pipeline: This module will initiate the training pipeline where each of the above mentioned components  
                      will be called sequentially untill model is saved.
2. Prediction Pipeline: This module will help getting prediction from saved trained model.

## Contributors
Shivansh Kaushal