import os
import sys
from src.logger import logging
from src.exception import CustomException
import pandas as pd

from sklearn.model_selection import train_test_split
from dataclasses import dataclass

'''
This decorator is used to remove the need of initiating a class __init__
SO, using this you can just star defining/creating variables without defining the class
'''
@dataclass
class DataIngestionConfig:
    train_data_path:str=os.path.join('artifact',"train.csv")
    test_data_path:str=os.path.join('artifact',"test.csv")
    raw_data_path: str = os.path.join('artifact', "data.csv")

class DataIngestion:
    def __init__(self):
        self.ingestion_config=DataIngestionConfig() #this variable will store all data paths defined in DataIngestionConfig()

    def initiate_data_ingestion(self):
        logging.info("Entered the ")
