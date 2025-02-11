import os
import sys
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
from sklearn.linear_model import SGDRegressor
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
                "SGD Regressor": SGDRegressor(),
                "K-Neighbors Regressor": KNeighborsRegressor(),
                "XG-Boost Regressor": XGBRegressor(),
                "CatBoosting Regressor": CatBoostRegressor(verbose=False),
                "AdaBoost Regressor": AdaBoostRegressor(),
            }

            params = {
                "Decision Tree": {
                    'criterion': ['squared_error', 'friedman_mse', 'absolute_error', 'poisson'],
                    'max_depth': [2,3,4]
                    # 'splitter':['best','random'],
                    # 'max_features':['sqrt','log2'],
                },
                "Random Forest": {
                    # 'criterion':['squared_error', 'friedman_mse', 'absolute_error', 'poisson'],

                    # 'max_features':['sqrt','log2',None],
                    'n_estimators': [8, 16, 32, 64, 128, 256]
                },
                "Gradient Boosting": {
                    # 'loss':['squared_error', 'huber', 'absolute_error', 'quantile'],
                    'learning_rate': [.1, .01, .05, .001],
                    'subsample': [0.6, 0.7, 0.75, 0.8, 0.85, 0.9],
                    # 'criterion':['squared_error', 'friedman_mse'],
                    # 'max_features':['auto','sqrt','log2'],
                    'n_estimators': [8, 16, 32, 64, 128, 256]
                },
                "Linear Regression": {},
                "SGD Regressor":{
                    'penalty':['l2', 'l1', 'elasticnet'],
                    'alpha':[0.0001, 0.0005,0.001,0.0015,0.0020]
                },
                "K-Neighbors Regressor": {
                    'n_neighbors': [3, 5, 7, 9, 12, 15],
                    'algorithm': ['auto', 'ball_tree', 'kd_tree', 'brute']
                },
                "XG-Boost Regressor": {
                    'learning_rate': [.1, .01, .05, .001],
                    'n_estimators': [8, 16, 32, 64, 128, 256]
                },
                "CatBoosting Regressor": {
                    'depth': [6, 8, 10],
                    'learning_rate': [0.01, 0.05, 0.1],
                    'iterations': [30, 50, 100]
                },
                "AdaBoost Regressor": {
                    'learning_rate': [.1, .01, 0.5, .001],
                    # 'loss':['linear','square','exponential'],
                    'n_estimators': [8, 16, 32, 64, 128, 256]
                }

            }

            logging.info("Evaluating Model...")
            model_report:dict=evaluate_models(X_train=X_train,y_train=y_train,X_test=X_test,y_test=y_test,
                                             models=models,param=params)

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


'''  
  "KNeighborsRegressor": {
      'n_neighbors':[3,5,7,9,12,15],
      'algorithm': ['auto', 'ball_tree', 'kd_tree', 'brute']
  }, 
'''