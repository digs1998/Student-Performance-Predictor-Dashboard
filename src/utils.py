'''
Use this script for
common functionalities to be used by all parts of codebase, like reading data from database,
save models to cloud
'''
import os
import sys

import numpy as np 
import pandas as pd 
from src.exception import CustomException
from src.logger import logging
import dill
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error

def save_object(file_path, obj):
    try:
        dir_path = os.path.dirname(file_path)
        os.makedirs(dir_path, exist_ok=True)
        
        with open(file_path, "wb") as file_obj:
            dill.dump(obj, file_obj)
            
    except Exception as e:
        raise CustomException(e,sys)
    
def evaluate_models(xtrain, ytrain, xtest, ytest, models):
    try:
        report = dict()
        
        for i in range(len(list(models))):
            
            model = list(models.values())[i]
            
            model.fit(xtrain, ytrain)
            
            ytrain_pred = model.predict(xtrain)
            ytest_pred = model.predict(xtest)
            
            train_model_score = r2_score(ytrain, ytrain_pred)
            test_model_score = r2_score(ytest, ytest_pred)
            
            report[list(models.keys())[i]] = test_model_score
        
        return report
            
    except Exception as e:
        raise CustomException(e, sys)