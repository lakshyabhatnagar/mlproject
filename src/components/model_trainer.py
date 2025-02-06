import os,sys
import numpy as np
import pandas as pd
from dataclasses import dataclass

from catboost import CatBoostRegressor
from sklearn.ensemble import AdaBoostRegressor, RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score, mean_squared_error
from sklearn.neighbors import KNeighborsRegressor
from sklearn.tree import DecisionTreeRegressor
from xgboost import XGBRegressor
from src.utils import evaluate_model,saved_obj

from src.exception import CustomException
from src.logger import logging

@dataclass
class ModelTrainerConfig:
    trained_model_file_path=os.path.join('artifacts','model.pkl')
    
class ModelTrainer:
    def __init__(self):
        self.model_trainer_config=ModelTrainerConfig()

    def initiateModelTraining(self,train_array,test_array):
        try:
            logging.info("Split train and test input data")
            X_train,y_train,X_test,y_test=(train_array[:,:-1],train_array[:,-1],test_array[:,:-1],test_array[:,-1])
            models={
                'LinearRegression':LinearRegression(),
                'DecisionTreeRegressor':DecisionTreeRegressor(),
                'RandomForestRegressor':RandomForestRegressor(),
                'AdaBoostRegressor':AdaBoostRegressor(),
                'GradientBoostingRegressor':GradientBoostingRegressor(),
                'KNeighborsRegressor':KNeighborsRegressor(),
                'XGBRegressor':XGBRegressor(),
                'CatBoostRegressor':CatBoostRegressor(verbose=False)
            }

            model_report:dict=evaluate_model(model=models,X_train=X_train,Y_train=y_train,X_test=X_test,Y_test=y_test)
            best_model_score=max(sorted(model_report.values()))
            best_model_name=[key for key in model_report if model_report[key]==best_model_score][0]
            best_model=models[best_model_name]
            if best_model_score<0.6:
                raise CustomException("No best model found",sys)
            logging.info(f"Best model found is {best_model_name} with score {best_model_score}")
            saved_obj(self.model_trainer_config.trained_model_file_path,best_model)
            pred=best_model.predict(X_test)
            r2_square=r2_score(y_test,pred)
            return r2_square
        except Exception as e:
            raise CustomException(e,sys)
        
            


