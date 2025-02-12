import os
import sys
from src.exception import CustomException
from src.logger import logging
import pandas as pd 

from sklearn.model_selection import train_test_split
from dataclasses import dataclass

@dataclass
class DataIngestionConfig:
    train_data_path: str=os.path.join("src/components/artifacts", "train.csv")
    test_data_path: str=os.path.join("src/components/artifacts", "test.csv")
    raw_data_path: str=os.path.join("src/components/artifacts", "data.csv")
    
class DataIngestion:
    def __init__(self):
        self.ingestion_config = DataIngestionConfig()
        
    def initiate_data_ingestion(self):
        '''
        use this function to read data from data sources
        '''
        logging.info("Initiated Data Ingestion Method!")
        try:
            df = pd.read_csv('/Users/Lenovo/Desktop/Education/starter-ml-project/notebooks/data/stud.csv')
            logging.info('---- Read the csv file, and loaded dataframe ----')
            
            os.makedirs(os.path.dirname(self.ingestion_config.train_data_path), exist_ok=True)
            
            df.to_csv(self.ingestion_config.raw_data_path, index=False, header=True)
            
            logging.info("---- Train Test split initiated ----")
            train_Set, test_Set = train_test_split(df, test_size=0.2, random_state=45)
            
            train_Set.to_csv(self.ingestion_config.train_data_path, index=False, header=True)
            test_Set.to_csv(self.ingestion_config.test_data_path, index=False, header=True)
            
            logging.info(" ----- Ingestion completed for the data ----")
            
            return (
                self.ingestion_config.train_data_path,
                self.ingestion_config.test_data_path,
            )
        except Exception as e:
            raise CustomException(e, sys)
        
if __name__ == "__main__":
    
    from src.components.data_transformation import DataTransformation
    from src.components.data_transformation import DataTransformationConfig
    from src.components.model_trainer import ModeTrainerConfig, ModelTrainer
    
    obj = DataIngestion()
    train_data, test_data = obj.initiate_data_ingestion()
    
    data_transformation = DataTransformation()
    train_array, test_array , _ = data_transformation.initiate_data_transformation(train_data, test_data)
    
    model_trainer = ModelTrainer()
    print(model_trainer.initiate_model_training(train_array=train_array, test_array=test_array))