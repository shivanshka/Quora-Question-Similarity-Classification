import os, sys
from zipfile import ZipFile
from urllib.request import urlretrieve
from Quora_App.logger import logging
from Quora_App.config import ConfigurationManager
from Quora_App.exception import ApplicationException
from Quora_App.entity import DataIngestionConfig
from Quora_App.entity import DataIngestionArtifact
from Quora_App.utils import create_directories
from Quora_App.constants import *
import pandas as pd
import numpy as np
from pathlib import Path

class DataIngestion:
    def __init__(self, data_ingestion_config: DataIngestionConfig):
        try:
            logging.info(f"\n{'>'*15} Data Ingestion Started {'<'*15}\n")
            self.data_ingestion_config= data_ingestion_config
        except Exception as e:
            raise ApplicationException(e, sys) from e

    def download_data(self):
        """
        Downloads the zipped dataset from the  specified url and save it to the specified path
        """
        try:
            # Extracting remote url to download dataset
            download_url = self.data_ingestion_config.download_url

            # folder loacation to save zipped data
            zipped_download_dir = self.data_ingestion_config.zipped_download_dir

            create_directories([zipped_download_dir])

            file_name = self.data_ingestion_config.download_file_name
            download_file_path = Path(zipped_download_dir,file_name)
            # Downloading data from url
            logging.info(f"Downloading file from {download_url}")
            urlretrieve(download_url, download_file_path)
            logging.info(f"File {file_name} has been downloaded successfully!!!")
            return download_file_path
        except Exception as e:
            raise ApplicationException(e, sys) from e
        
    def extract_zipped_data(self, zipped_file_path: Path):
        "Unzipped archive files and save it to raw directory"
        try:
            raw_data_dir = self.data_ingestion_config.raw_data_dir
            create_directories([raw_data_dir])

            logging.info(f"Extracting zipped file: [{zipped_file_path}] into dir: [{raw_data_dir}]")
            with ZipFile(zipped_file_path) as f:
                f.extractall(Path(raw_data_dir))
            logging.info("Unzipping compeleted sucessfully!!!!")

            data_ingestion_artifact= DataIngestionArtifact(is_ingested=True,
                                                           message="Data Ingestion Successfull")
            return data_ingestion_artifact
        except Exception as e:
            raise ApplicationException(e, sys) from e
        
    def initiate_data_ingestion(self)-> DataIngestionArtifact:
        try:
            zipped_file_path= self.download_data()
            return self.extract_zipped_data(Path(zipped_file_path))
        except Exception as e:
            raise ApplicationException(e, sys) from e  
        
    def __del__(self):
        logging.info(f"\n{'>'*15} Data Ingestion Completed {'<'*15}\n")
