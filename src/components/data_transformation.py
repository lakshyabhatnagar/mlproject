#for feature engineering 
import sys
from dataclasses import dataclass
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
import os
from src.exception import CustomException
from src.logger import logging
from src.utils import saved_obj

@dataclass
class DataTransformationConfig:
    preprocessor_obj_filepath=os.path.join('artifacts','preprocessor.pkl')

class DataTransformation:
    def __init__(self):
        self.data_transformation_config = DataTransformationConfig()

    def get_data_transformation_object(self): 
        #resposnsible for data transformation
        try:
            numerical_features = ['reading_score', 'writing_score']
            categorical_features = ['gender','race_ethnicity','parental_level_of_education','lunch','test_preparation_course']  
            num_pipeline = Pipeline(steps=[
                ('imputer', SimpleImputer(strategy='median')),
                ('scaler', StandardScaler())])
            cat_pipeline = Pipeline(steps=[
                ('imputer', SimpleImputer(strategy='most_frequent')),
                ('onehot', OneHotEncoder(handle_unknown='ignore')),
                ('scaler', StandardScaler(with_mean=False))])
            logging.info('Numerical Features: {}'.format(numerical_features))
            logging.info('Categorical Features: {}'.format(categorical_features))
            preprocessor = ColumnTransformer(
                transformers=[
                    ('num_pipeline', num_pipeline, numerical_features),
                    ('cat_pipeline', cat_pipeline, categorical_features)])
            return preprocessor
        except Exception as e:
            raise CustomException(e,sys)
    def initiateDataTransformation(self,train_path,test_path):
        try:
            train_df=pd.read_csv(train_path)
            test_df=pd.read_csv(test_path)
            logging.info('Train and test data read')
            logging.info('obtaining preprocessor object')
            preprocessor_obj=self.get_data_transformation_object()
            target_column_name='math_score'
            input_feature_train_df=train_df.drop(target_column_name,axis=1)
            target_train_df=train_df[target_column_name]

            input_feature_test_df=test_df.drop(target_column_name,axis=1)
            target_test_df=test_df[target_column_name]
            logging.info('Fitting the preprocessor object')
            input_feature_train_arr=preprocessor_obj.fit_transform(input_feature_train_df)
            input_feature_test_arr=preprocessor_obj.transform(input_feature_test_df)
            train_arr=np.c_[input_feature_train_arr,np.array(target_train_df)]
            test_arr=np.c_[input_feature_test_arr,np.array(target_test_df)]
            logging.info('saved preprocessor object')
            saved_obj(file_path=self.data_transformation_config.preprocessor_obj_filepath, obj=preprocessor_obj)
            return (train_arr,test_arr,self.data_transformation_config.preprocessor_obj_filepath)
        except Exception as e:
            raise CustomException(e,sys)





