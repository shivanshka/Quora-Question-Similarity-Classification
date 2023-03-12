from Quora_App.pipeline import Training_Pipeline
from Quora_App.logger import logging

train= Training_Pipeline()

if __name__=="__main__":
    train.run_training_pipeline()
    