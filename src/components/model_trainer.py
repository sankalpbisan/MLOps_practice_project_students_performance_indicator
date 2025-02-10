import os
from dataclasses import dataclass
from catboost import CatBoostRegressor
from sklearn.ensemble import (
AdaBoostRegressor,
GradientBoostingRegressor,
RandomForestRegressor)
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
from sklearn.neighbors import KNeighborsRegressor
from sklearn.tree import DecisionTreeRegressor
from xgboost import XGBRegressor
import sys

from src.logger import logging
from src.exception import CustomException
from src.utils import proj_dir_path,evaluate_models,save_object

@dataclass
class ModelTrainerConfig:
    trained_model_file_path = os.path.join(proj_dir_path+"artifact","model.pkl")

class ModelTrainer:
    def __init__(self):
        self.model_trainer_config = ModelTrainerConfig()

    def initiate_model_trainer(self,train_arr,test_arr):
        try:
            logging.info("Splitting training and testing data")
            X_train, y_train, X_test, y_test = (
                train_arr[:,:-1],   #[all:rows,1st_col:last_col(excluding)]
                train_arr[:,-1],  #[all:rows,last_col(only)]
                test_arr[:, :-1],   #[all:rows,1st_col:last_col(excluding)]
                test_arr[:,-1],   #[all:rows,last_col(only)]
            )

            models = {
                "Random Forest": RandomForestRegressor(),
                "Decision Tree": DecisionTreeRegressor(),
                "Gradient Boosting": GradientBoostingRegressor(),
                "Linear Regression": LinearRegression(),
                "K-Neighbors Classifier": KNeighborsRegressor(),
                "XGBClassifier": XGBRegressor(),
                "CatBoosting Classifier": CatBoostRegressor(verbose=False),
                "AdaBoost Classifier": AdaBoostRegressor(),
            }

            logging.info("Evaluating Model...")
            model_report:dict=evaluate_models(X_train=X_train,y_train=y_train,X_test=X_test,y_test=y_test,
                                             models=models)

            logging.info("Evaluation Completed. Now, Finding the best model")

            #To get Best model among all models
            best_model_score = max(sorted(model_report.values()))
            ## To get best model name from dict
            best_model_name = list(model_report.keys())[
                list(model_report.values()).index(best_model_score)
            ]
            best_model = models[best_model_name]
            logging.info(f"Found it: {best_model} is the Best model")

            if best_model_score<0.6:
                raise CustomException("No best model found")

            logging.info("Saving the model in to pickle file")

            save_object(
                file_path = self.model_trainer_config.trained_model_file_path,
                obj=best_model
            )

            logging.info("Pickle file saved successfully")

            predicted = best_model.predict(X_test)
            r2 = r2_score(y_test,predicted)

            logging.info(f"Best Model score {r2}")

            return r2

        except Exception as e:
            raise CustomException(e,sys)

