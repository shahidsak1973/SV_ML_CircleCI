import os
import pandas
from src.logger import get_logger
from src.custom_exception import CustomException
import yaml
import pandas as pd
import sys

logger = get_logger(__name__)

def read_yaml(file_path):
    try:
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"YAML file not found at path: {file_path}")
        
        with open(file_path, "r") as yaml_file:
            config = yaml.safe_load(yaml_file)
            logger.info(f"Successfully read YAML file from {file_path}")
            return config

    except Exception as e:
        logger.error(f"Error while reading YAML file from {file_path}")
        raise CustomException(f"Failed to read YAML file at path: {file_path}",sys)


def load_data(path):
    try:
        logger.info("Loading data")
        return pd.read_csv(path)
    except Exception as e:
        logger.error(f"Error loading the data {e}")
        raise CustomException("Failed to load data" , e)
    