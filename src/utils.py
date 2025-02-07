import os
import sys
from src.exception import CustomException
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
import dill
from sklearn.metrics import r2_score, mean_squared_error
def saved_obj(file_path,obj):
    try:
        dirpath=os.path.dirname(file_path)
        os.makedirs(dirpath, exist_ok=True)
        with open(file_path, 'wb') as f:
            dill.dump(obj, f)
        return True
    except Exception as e:
        raise CustomException(e,sys)
def evaluate_model(model:dict,X_train:np.array,Y_train:np.array,X_test:np.array,Y_test:np.array)->dict:
    try:
        model_report={}
        for i in range(len(list(model))):
            models=list(model.values())[i]
            models.fit(X_train,Y_train)
            y_train_pred=models.predict(X_train)
            y_test_pred=models.predict(X_test)
            train_model_score=r2_score(Y_train,y_train_pred)
            test_model_score=r2_score(Y_test,y_test_pred)
            model_report[list(model.keys())[i]]=test_model_score
        return model_report

    except Exception as e:
        raise CustomException(e,sys)
    
def load_object(file_path):
    try:
        with open(file_path,"rb") as f:
            return dill.load(f)
    except Exception as e:
        raise CustomException(e,sys)
    