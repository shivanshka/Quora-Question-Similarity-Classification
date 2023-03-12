import os
import yaml
from Quora_App.logger import logging
import json
import pickle
from ensure import ensure_annotations
from box import ConfigBox
from box.exceptions import BoxValueError
import pandas as pd
from pathlib import Path


@ensure_annotations
def read_yaml(path_to_yaml:Path) -> ConfigBox:
    """
    Reads YAML file and returns
    Args:
        path_to_yaml (str): path like input
    Raises:
        ValueError: if yaml file is empty
        e: empty file
    Returns:
        ConfigBox: ConfigBox type
    """
    try:
        with open(path_to_yaml) as yaml_file:
            content = yaml.safe_load(yaml_file)
            logging.info(f"yaml file: {path_to_yaml} loaded successfully")
            return ConfigBox(content)
    except BoxValueError:
        raise ValueError("yaml file is empty")
    except Exception as e:
        raise e

@ensure_annotations
def create_directories(path_to_directories:list,verbose=True):
    """
    Create list of directories
    Args:
        path_to_directories (list): list of path of directories
        ignore_log (bool, optional): ignore of multiple dirs is to be created. Defaults to False
    """
    for path in path_to_directories:
        os.makedirs(path, exist_ok=True)
        if verbose:
            logging.info(f"Created directory at: {path}")

@ensure_annotations
def save_json(path: Path,data:dict):
    """
    Save json data
    Args:
        path (Path): path to json file
        data (dict): data to be saved in json file
    """

    with open(path, "w") as f:
        json.dump(data,f,indent=4)

    logging.info(f"json file saved at {path}")

@ensure_annotations
def load_json(path: Path) -> ConfigBox:
    """
    Load json files data
    Args:
        path (Path): path to json file
    Returns:
        ConfigBox: data as class attributes instead of dict
    """

    with open(path) as f:
        content = json.load(f)

    logging.info(f"json file loaded successfully from: {path}")
    return ConfigBox(content)

@ensure_annotations
def save_bin(obj, path: Path):
    """
    Save binary file
    Args:
        obj: object to be saved as binary
        path (Path): path to binary file
    """
    with open(path,"wb") as file:
        pickle.dump(file=file, obj=obj)
    logging.info(f"Binary file saved at: {path}")

@ensure_annotations
def load_bin(path: Path):
    """
    Load Binary Data
    Args:
        path (Path): path to binary file
    Returns:
        Any: object stored in the file
    """
    with open(path,"rb") as file:
        data = pickle.load(file)
    logging.info(f"binary file loaded from: {path}")
    return data

@ensure_annotations
def get_size(path: Path) -> str:
    """
    Get size in KB
    Args:
        path (Path): path of the file
    Returns:
        str: size in KB
    """
    size_in_kb = round(os.path.getsize(path)/1024)
    return f"~{size_in_kb} KB"

@ensure_annotations
def read_data(filepath:Path)-> pd.DataFrame:
    """
    Reads the data from specified file path and returns pandas dataframe
    """
    if str(filepath).endswith(".csv"):
        data = pd.read_csv(filepath)
        return data

    elif str(filepath).endswith(".parquet"):
        data = pd.read_parquet(filepath)
        return data
    
    else:
        logging.error(f"Error occured during reading data from {filepath}")

@ensure_annotations
def save_data(filepath:Path, df:pd.DataFrame, format:str='csv'):
    """
    Saves the dataframe to specified file path either in .CSV or .PARQUET format
     Params:
        format (str) : 'csv' for saving in .csv format (default)
                       'parquet' for saving in .parquet format
    """
    if format=='csv':
        df.to_csv(filepath, index=None)

    elif format=='parquet':
        df.to_parquet(filepath)
    
    else:
        logging.error(f"Error occured during saving data to {filepath}")
    
    