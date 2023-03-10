import os, sys
import re
import pandas as pd
from pathlib import Path
from Quora_App.constants import *
from Quora_App.utils import read_yaml, read_data
from Quora_App.logger import logging
from Quora_App.exception import ApplicationException
from Quora_App.entity import DataIngestionConfig, DataValidationConfig, DataValidationArtifact

class DataValidation:
    def __init__(self, data_validation_config: DataValidationConfig,
                 data_ingestion_config: DataIngestionConfig):
        try:
            logging.info(f"\n{'>'*15} Data Validation Started {'<'*15}\n")
            self.data_validation_config = data_validation_config
            self.data_ingestion_config = data_ingestion_config
            self.schema_file_path = self.data_validation_config.schema_file_path
            self.schema = read_yaml(self.schema_file_path)
        except Exception as e:
            raise ApplicationException(e, sys) from e

    def column_check(self, file: Path, train:bool)->bool:
        try:
            features_count = self.schema.Features_Count
            features = list(self.schema.Features.keys())

            if train==True:
                features_count = features_count+1
                features.append(self.schema.target_feature)

            data = read_data(file)
            # Finding no of columns in dataset
            no_of_columns= data.shape[1]

            #Checking if no of columns is according to given schema
            if no_of_columns != features_count:
                raise Exception(f"No of columns is not correct in file: [{file}]")
            logging.info(f"Is features count:[{features_count}] correct: True")

            # Checking features name,  whether they are as per the given schema
            for feature in data.columns:
                if feature not in features:
                    raise Exception(f"Feature: [{feature}] in the file: [{file}] not available in given schema")
            logging.info(f"Are all feature Names correct: True")

            # Checking datatype of features
            columns = list(data.columns)
            columns.remove(self.schema.target_feature)
            for feature in columns:
                if data[feature].dtype != self.schema.Features[feature]:
                    raise Exception(f"Feature: [{feature}] datatype is not as per given schema") 
            logging.info(f"Are all feature datatype correct: True")

            # Checking whether any column have entire rows as missing value
            count = 0
            col = []
            for column in data.columns:            
                if (len(data[column]) - data[column].count()) == len(data[column]):
                    count+=1
                    col.append(column)
            if count > 0:
                raise Exception(f"Columns: [{col}] have entire row as missing value") 
            logging.info("Does any feature have entire rows as null values: False")
            return True
        except Exception as e:
            raise ApplicationException(e, sys) from e     
          
    def validate_dataset_schema(self,train:bool)->bool:
        try:
            logging.info("Validating the schema of the dataset")
            validation_status=False

            raw_data_dir = self.data_ingestion_config.raw_data_dir

            for file in os.listdir(raw_data_dir):
                if self.column_check(Path(raw_data_dir, file), train):
                    validation_status=True

            logging.info("Schema Validation Completed")
            return validation_status
        except Exception as e:
            raise ApplicationException(e, sys) from e
        
    def initiate_data_validation(self, train:bool) -> DataValidationArtifact:
        try:
            self.validate_dataset_schema(train)
            data_validation_artifact = DataValidationArtifact(schema_file_path= self.data_validation_config.schema_file_path,
                                                              is_validated=True,
                                                              message="Data Validation Successful!!!")
            logging.info(f"Data Validation Artifact: {data_validation_artifact}")
            return data_validation_artifact
        except Exception as e:
            raise ApplicationException(e, sys) from e
        
    def __del__(self):
        logging.info(f"\n{'>'*15} Data Validation Completed {'<'*15}\n")
