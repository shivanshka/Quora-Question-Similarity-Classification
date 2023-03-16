import os, sys
from pathlib import Path
from Quora_App.logger import logging
from Quora_App.exception import ApplicationException
from Quora_App.entity import DataIngestionConfig, DataValidationConfig, DataTransformationConfig
from Quora_App.entity import DataIngestionArtifact, DataValidationArtifact, DataTransformationArtifact, ModelTrainerArtifact
from Quora_App.components import DataIngestion, DataValidation, DataTransformation, ModelTrainer
from Quora_App.constants import *
from Quora_App.config.configuration2 import Configuration2

class Training_Pipeline2:
    def __init__(self, config: Configuration2= Configuration2())->None:
        try:
            logging.info(f"\n{'>'*20} Initiating Training Pipeline {'<'*20}\n")
            self.config=config
        except Exception as e:
            raise ApplicationException(e, sys) from e
        
    def start_data_ingestion(self, data_ingestion_config: DataIngestionConfig)-> DataIngestionArtifact:
        try:
            data_ingestion= DataIngestion(data_ingestion_config=data_ingestion_config)

            if os.path.exists(data_ingestion_config.raw_data_dir):
                logging.info("Data exists!!! ....Skipping Stage Data Ingestion")
                data_ingestion_artifact= DataIngestionArtifact(is_ingested=True,message="already available",
                                                               data_file_path=data_ingestion_config.raw_data_dir)
                return data_ingestion_artifact
            else:
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
            data_transformation_config= self.config.get_data_transformation_config()
            data_transformation=  DataTransformation(data_transformation_config= data_transformation_config,
                                                     data_ingestion_artifact=data_ingestion_artifact)
            if os.path.exists(data_transformation_config.transformed_train_dir):
                logging.info("Files exists.....Skipping Data Transformation Stage")
                data_transformation_artifacts= DataTransformationArtifact(is_transformed=True,
                                                        message="Data Transfromation already exists",
                                                        transformed_train_file_path=Path(data_transformation_config.transformed_train_dir,"transformed_train.parquet"),
                                                        transformed_test_file_path=Path(data_transformation_config.transformed_test_dir,"transformed_test.parquet"),
                                                        feat_eng_obj_file_path=Path(data_transformation_config.feature_eng_object_file_path),
                                                        preprocessed_obj_file_path=Path(data_transformation_config.preprocessed_object_file_path),
                                                        word2tfidf_object_file_path=Path(data_transformation_config.word2tfidf_object_file_path))
                return data_transformation_artifacts
            else:
                return data_transformation.initiate_data_transformation()
        except Exception as e:
            raise ApplicationException(e, sys) from e
        
    def start_model_training(self, data_transformation_artifact:DataIngestionArtifact)->ModelTrainerArtifact:
        try:
            model_trainer=  ModelTrainer(model_trainer_config= self.config.get_model_trainer_config(),
                                               data_transformation_artifact=data_transformation_artifact)
            return model_trainer.initiate_model_training()
        
        except Exception as e:
            raise ApplicationException(e, sys) from e
        
    def run_training_pipeline(self):
        try:
            data_ingestion_config= self.config.get_data_ingestion_config()
            data_ingestion_artifact= self.start_data_ingestion(data_ingestion_config=data_ingestion_config)
            #data_validation_artifact= self.start_data_validation(data_ingestion_config=data_ingestion_config)
            data_transformation_artifact= self.start_data_transformation(data_ingestion_artifact=data_ingestion_artifact)
            model_trainer_artifact= self.start_model_training(data_transformation_artifact=data_transformation_artifact)
        except Exception as e:
            raise ApplicationException(e, sys) from e
        
    #def __del__(self):
    #    logging.info(f"\n{'>'*20} Training Pipeline Complete {'<'*20}\n")