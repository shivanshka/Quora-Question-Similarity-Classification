import os
from Quora_App.constants import *
from Quora_App.exception import ApplicationException
from Quora_App.logger import logging
from Quora_App.utils import read_yaml
from Quora_App.entity import DataIngestionConfig, DataValidationConfig, DataTransformationConfig, \
                            ModelTrainerConfig, ModelEvaluationConfig, TrainingPipelineConfig
import sys
from pathlib import Path

class ConfigurationManager:
    def __init__(self, config_file_path:str= CONFIG_FILE_PATH, 
                 timestamp:str= CURRENT_TIME_STAMP)-> None:
        try:
            self.config = read_yaml(Path(config_file_path))
            self.timestamp= timestamp
            self.training_pipeline_config = self.get_training_pipeline_config()
            self.artifact_dir= self.training_pipeline_config.artifact_dir
        except Exception as e:
            raise ApplicationException(e, sys) from e
        
    def get_data_ingestion_config(self) -> DataIngestionConfig:
        try:
            data_ingestion_info = self.config.data_ingestion_config
            data_ingestion_artifact_dir = os.path.join(self.artifact_dir,
                                                       DATA_INGESTION_ARTIFACT_DIR_KEY,
                                                       self.timestamp)
            download_url=data_ingestion_info.download_url

            zipped_download_dir= os.path.join(data_ingestion_artifact_dir,
                                              data_ingestion_info.zipped_download_dir)
            
            raw_data_dir= os.path.join(data_ingestion_artifact_dir,
                                              data_ingestion_info.raw_data_dir)
            
            download_file_name= data_ingestion_info.download_file_name
            data_ingestion_config= DataIngestionConfig(raw_data_dir=Path(raw_data_dir),
                                                       download_url=download_url,
                                                       zipped_download_dir=Path(zipped_download_dir),
                                                       download_file_name=download_file_name)
            
            logging.info(f"Data Ingestion Config: {data_ingestion_config}")
            return data_ingestion_config
        except Exception as e:
            raise ApplicationException(e, sys) from e
        
    def get_data_validation_config(self) -> DataValidationConfig:
        try:
            data_validation_info = self.config.data_validation_config
            data_validation_artifact_dir = os.path.join(self.artifact_dir,
                                                        DATA_VALIDATION_ARTIFACT_DIR_KEY,
                                                        self.timestamp)
            schema_file_path= os.path.join(ROOT_DIR,
                                           data_validation_info.schema_dir,
                                           data_validation_info.schema_file_name)
            
            data_validation_config= DataValidationConfig(schema_file_path=Path(schema_file_path))
            logging.info(f"Data Validation Config: {data_validation_config}")
            return data_validation_config
        except Exception as e:
            raise ApplicationException(e, sys) from e
        
    def get_data_transformation_config(self) -> DataTransformationConfig:
        try:
            data_transformation_info= self.config.data_transformation_config
            data_transformation_artifact_dir = os.path.join(self.artifact_dir,
                                                            DATA_TRANSFORMATION_ARTIFACT_DIR_KEY,
                                                            self.timestamp)
            
            transformed_train_dir= os.path.join(data_transformation_artifact_dir,
                                                data_transformation_info.transformed_dir,
                                                data_transformation_info.transformed_train_dir)
            
            transformed_test_dir= os.path.join(data_transformation_artifact_dir,
                                                data_transformation_info.transformed_dir,
                                                data_transformation_info.transformed_test_dir)
            
            feature_eng_object_file_path= os.path.join(data_transformation_artifact_dir,
                                                data_transformation_info.preprocessed_dir,
                                                data_transformation_info.feature_eng_object_file_name)
            
            preprocessed_object_file_path= os.path.join(data_transformation_artifact_dir,
                                                data_transformation_info.preprocessed_dir,
                                                data_transformation_info.preprocessed_object_file_name)
            
            data_transformation_config= DataTransformationConfig(transformed_train_dir=Path(transformed_train_dir),
                                                                 transformed_test_dir=Path(transformed_test_dir),
                                                                 feature_eng_object_file_path=Path(feature_eng_object_file_path),
                                                                 preprocessed_object_file_path=Path(preprocessed_object_file_path))
            logging.info(f"Data Transformation Config: {data_transformation_config}")
            return data_transformation_config
        except Exception as e:
            raise ApplicationException(e, sys) from e
        
    def get_model_trainer_config(self) -> ModelTrainerConfig:
        try:
            model_trainer_info= self.config.model_trainer_config
            model_trainer_artifact_dir= os.path.join(self.artifact_dir,
                                                     MODEL_TRAINER_ARTIFACT_DIR_KEY,
                                                     self.timestamp)
            
            trained_model_file_path= os.path.join.path(model_trainer_artifact_dir,
                                                       model_trainer_info.model_file_name)
            
            model_trainer_config= ModelTrainerConfig(trained_model_file_path=Path(trained_model_file_path),
                                                     base_accuracy=model_trainer_info.base_accuracy)
            
            logging.info(f"Model Trainer Config: {model_trainer_config}")
            return model_trainer_config
        except Exception as e:
            raise ApplicationException(e, sys) from e
        
    def get_training_pipeline_config(self) -> TrainingPipelineConfig:
        try:
            training_pipeline_config= self.config.training_pipeline_config
            artifact_dir = os.path.join(ROOT_DIR, 
                                        training_pipeline_config.pipeline_name,
                                        training_pipeline_config.artifact_dir)
            
            training_pipeline_config = TrainingPipelineConfig(artifact_dir= Path(artifact_dir))
            logging.info(f"Training Pipeline Config: {training_pipeline_config}")
            return training_pipeline_config
        except Exception as e:
            raise ApplicationException(e, sys) from e