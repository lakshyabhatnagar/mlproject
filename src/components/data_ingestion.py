import os, sys
from src.exception import CustomException
from src.logger import logging
import pandas as pd
from sklearn.model_selection import train_test_split
from dataclasses import dataclass
from src.components.data_transformation import DataTransformation
from src.components.data_transformation import DataTransformationConfig
from src.components.model_trainer import ModelTrainer
from src.components.model_trainer import ModelTrainerConfig

@dataclass
class DataIngestionConfig:
    train_data_path: str=os.path.join('artifacts',"train.csv")
    test_data_path: str=os.path.join('artifacts',"test.csv")
    raw_data_path: str=os.path.join('artifacts',"data.csv")

class DataIngestion:
    def __init__(self):
        self.ingestion_config=DataIngestionConfig()
    def initiate_data_ingestion(self):                    #to read data from database
        logging.info("entered data ingestion method")
        try:
            df=pd.read_csv('notebook\data\stud.csv')
            logging.info("data read successfully")
            os.makedirs(os.path.dirname(self.ingestion_config.train_data_path), exist_ok=True)
            df.to_csv(self.ingestion_config.raw_data_path, index=False,header=True)
            logging.info('Train Test split intiated')
            train_set,test_set=train_test_split(df,test_size=0.2,random_state=42)
            train_set.to_csv(self.ingestion_config.train_data_path, index=False,header=True)
            test_set.to_csv(self.ingestion_config.test_data_path, index=False,header=True)
            logging.info('Ingestion Completed')
            return(
                self.ingestion_config.train_data_path,
                self.ingestion_config.test_data_path,
            )
        except Exception as e:
            raise CustomException(e,sys)
            logging.error("Data Ingestion Failed")


if __name__ == "__main__":
    data_ingestion=DataIngestion()
    train_data,test_data=data_ingestion.initiate_data_ingestion()
    data_transformation=DataTransformation()
    data_transformation.initiateDataTransformation(train_data,test_data)
    train_arr,test_arr,_=data_transformation.initiateDataTransformation(train_data,test_data)

    modeltrainer=ModelTrainer()
    print(modeltrainer.initiateModelTraining(train_arr,test_arr))
    


