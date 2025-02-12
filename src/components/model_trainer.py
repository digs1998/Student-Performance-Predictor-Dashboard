import os
import sys
from dataclasses import dataclass

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt 
import seaborn as sns
# Modelling
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.neighbors import KNeighborsRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor,AdaBoostRegressor
from sklearn.svm import SVR
from sklearn.linear_model import LinearRegression, Ridge,Lasso
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
from sklearn.model_selection import RandomizedSearchCV
from catboost import CatBoostRegressor
import xgboost
import warnings

from src.exception import CustomException
from src.logger import logging
from src.utils import save_object, evaluate_models

@dataclass
class ModeTrainerConfig:
    trained_model_filepath = os.path.join("src/components/artifacts", "model.pkl")
    
class ModelTrainer:
    def __init__(self):
        self.model_trainer_config = ModeTrainerConfig()
        
    
    def initiate_model_training(self, train_array, test_array):
        try:
            logging.info("----- Splitting Training and Test input data -----")
            xtrain, ytrain, xtest, ytest = (
                train_array[:,:-1], 
                train_array[:,-1],
                test_array[:,:-1],
                test_array[:,-1]
            )
            
            model_dict = {
                "Random Forest": RandomForestRegressor(),
                "KNN": KNeighborsRegressor(),
                "XGBoost": xgboost.XGBRegressor(),
                "Adaboost": AdaBoostRegressor(),
                "Decision Trees": DecisionTreeRegressor(),
                "Linear Regression": LinearRegression(),
                "Catboost": CatBoostRegressor(verbose=False)
            }
            
            model_report = {}
            model_report = evaluate_models(xtrain=xtrain, ytrain=ytrain, 
                                          xtest=xtest, ytest=ytest, models=model_dict)
            
            best_model_score = max(model_report.values())  # Find the highest score
            best_model_key = max(model_report, key=model_report.get)  # Find corresponding key

            best_model = model_dict[best_model_key]  # Retrieve the model
            
            if best_model_score < 0.6:
                raise CustomException("No Best Models yet!")
            logging.info("--- Training Completed, Best Model found -----")
            
            save_object(
                file_path = self.model_trainer_config.trained_model_filepath,
                obj = best_model
            )
            
            predicted = best_model.predict(xtest)
            r2_score_model = r2_score(ytest, predicted)
            
            return r2_score_model
            
        except Exception as e:
            raise CustomException(e, sys)