import os
import sys
from src.exception import CustomException
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
import dill
def saved_obj(file_path,obj):
    try:
        dirpath=os.path.dirname(file_path)
        os.makedirs(dirpath, exist_ok=True)
        with open(file_path, 'wb') as f:
            dill.dump(obj, f)
        return True
    except Exception as e:
        raise CustomException(e,sys)
        return False
