import os
import sys

from src.exception import CustomException
from src.logger import logging
import pandas as pd

from sklearn.model_selection import train_test_split
from dataclasses import dataclass

from src.components.data_transformation import DataTransformation
from src.components.data_transformation import DataTransformationConfig

from src.components.model_trainer import ModelTrainerConfig
from src.components.model_trainer import ModelTrainer

from src.utils import proj_dir_path

'''
This decorator is used to remove the need of initiating a class __init__
SO, using this you can just star defining/creating variables without defining the class
'''
@dataclass
class DataIngestionConfig:
    train_data_path:str=os.path.join(proj_dir_path+'artifact',"train.csv")
    test_data_path:str=os.path.join(proj_dir_path+'artifact',"test.csv")
    raw_data_path: str=os.path.join(proj_dir_path+'artifact', "data.csv")

class DataIngestion:
    def __init__(self):
        self.ingestion_config=DataIngestionConfig() #this variable will store all data paths defined in DataIngestionConfig()

    def initiate_data_ingestion(self):
        logging.info("Entered the Data_Ingestion method (or component)")
        try:
            df = pd.read_csv('D:\\_Folder\\Pycharm_Projects_Dir\\MLOps_project\\notebook\\data\\stud.csv')
            logging.info("Data has been read and stored as DataFrame")

            os.makedirs(proj_dir_path+'artifact', exist_ok=True)
            df.to_csv(self.ingestion_config.raw_data_path,index=False,header=True)

            logging.info("Train-Test split Initiated")
            train_set,test_set=train_test_split(df,test_size=0.2,random_state=42)

            train_set.to_csv(self.ingestion_config.train_data_path,index=False,header=True)
            test_set.to_csv(self.ingestion_config.test_data_path, index=False, header=True)
            logging.info("Train_Test split completed and stored in artifact directory")

            logging.info("Data Ingestion Completed")

            return(
                self.ingestion_config.train_data_path,
                self.ingestion_config.test_data_path
            )

        except Exception as e:
            raise CustomException(e,sys)

if __name__ == "__main__":
    obj = DataIngestion()
    obj.initiate_data_ingestion()

#For running data_transformation.py & model_trainer.py uncomment following doc string
# '''
    train_data, test_data = obj.initiate_data_ingestion()

    data_transformation = DataTransformation()
    train_arr,test_arr,temp_var = data_transformation.initiate_data_transformation(train_data, test_data)

    modeltrainer = ModelTrainer()
    print(modeltrainer.initiate_model_trainer(train_arr,test_arr))

    
# '''

    # transform_obj = DataTransformation()
    #
    # transform_obj.initiate_data_transformation(train_path,test_path)

