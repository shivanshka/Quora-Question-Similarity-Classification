import os, sys
from Quora_App.logger import logging
from Quora_App.exception import ApplicationException
from Quora_App.entity import DataIngestionConfig, DataValidationConfig, DataTransformationConfig
from Quora_App.entity import DataIngestionArtifact, DataValidationArtifact, DataTransformationArtifact
from Quora_App.components import DataIngestion, DataValidation, DataTransformation
from Quora_App.constants import *
from Quora_App.config import ConfigurationManager

class Training_Pipeline:
    def __init__(self, config: ConfigurationManager= ConfigurationManager())->None:
        try:
            logging.info(f"\n{'>'*20} Initiating Training Pipeline {'<'*20}\n")
            self.config=config
        except Exception as e:
            raise ApplicationException(e, sys) from e
        
    def start_data_ingestion(self, data_ingestion_config: DataIngestionConfig)-> DataIngestionArtifact:
        try:
            data_ingestion= DataIngestion(data_ingestion_config=data_ingestion_config)
            return data_ingestion.initiate_data_ingestion()
        except Exception as e:
            raise ApplicationException(e, sys) from e
        
    def start_data_validation(self, data_ingestion_config: DataIngestionConfig)-> DataValidationArtifact:
        try:
            data_validation = DataValidation(data_validation_config=self.config.get_data_validation_config(),
                                             data_ingestion_config=data_ingestion_config)
            return data_validation.initiate_data_validation(train=True)
        except Exception as e:
            raise ApplicationException(e, sys) from e
        
    def start_data_transformation(self, data_ingestion_artifact:DataIngestionArtifact)-> DataTransformationArtifact:
        try:
            #
            data_transformation=  DataTransformation(data_transformation_config= self.config.get_data_transformation_config(),
                                                     data_ingestion_artifact=data_ingestion_artifact)
            return data_transformation.initiate_data_transformation()
        except Exception as e:
            raise ApplicationException(e, sys) from e
        
    def run_training_pipeline(self):
        try:
            data_ingestion_config= self.config.get_data_ingestion_config()
            data_ingestion_artifact= self.start_data_ingestion(data_ingestion_config=data_ingestion_config)
            data_validation_artifact= self.start_data_validation(data_ingestion_config=data_ingestion_config)
            data_transformation_artifact= self.start_data_transformation(data_ingestion_artifact=data_ingestion_artifact)
        except Exception as e:
            raise ApplicationException(e, sys) from e
        
    #def __del__(self):
    #    logging.info(f"\n{'>'*20} Training Pipeline Complete {'<'*20}\n")