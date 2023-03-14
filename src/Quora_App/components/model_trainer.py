import os, sys
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from datetime import datetime
from Quora_App.entity import ModelTrainerConfig, DataTransformationArtifact, DataValidationArtifact, ModelTrainerArtifact
from Quora_App.constants import *
from Quora_App.logger import logging
from Quora_App.exception import ApplicationException
from Quora_App.utils import save_bin, read_yaml, read_data, create_directories
import optuna
from xgboost import XGBClassifier
from sklearn.metrics import log_loss, roc_auc_score, confusion_matrix, roc_curve
from sklearn.model_selection import GridSearchCV
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.calibration import CalibratedClassifierCV

class ModelTrainer:
    def __init__(self, model_trainer_config: ModelTrainerConfig,
                       data_transformation_artifact: DataTransformationArtifact):
        try:
            logging.info(f"\n{'>'*20} Model Training Started {'<'*20}\n")
            self.model_trainer_config= model_trainer_config
            self.data_transformation_artifact= data_transformation_artifact
        except Exception as e:
            raise ApplicationException(e, sys) from e

    def get_best_params_xgboost(self, X_train, y_train, X_test, y_test)-> dict:
        try:
            logging.info("Running Hyperparameter Tuning for XGBoost Classifier")
            def objective(trial,data = X_train, target = y_train):
                param = {
                    'reg_lambda' : trial.suggest_loguniform('lambda', 1e-4, 10.0),
                    'reg_alpha' :  trial.suggest_loguniform('alpha', 1e-4, 10.0),
                    'booster' : trial.suggest_categorical('booster',['gbtree','dart']),
                    'colsample_bytree' : trial.suggest_categorical('colsample_bytree', [.1,.2,.3,.4,.5,.6,.7,.8,.9,1]),
                    'subsample' : trial.suggest_categorical('subsample', [.1,.2,.3,.4,.5,.6,.7,.8,.9,1]),
                    'learning_rate' : trial.suggest_categorical('learning_rate',[.00001,.0003,.008,.02,.01,1,10,20]),
                    'n_estimators' : trial.suggest_categorical('n_estimator',[200,300,400,500]),
                    'max_depth' : trial.suggest_categorical('max_depth', [3,4,5,6,7,8,9,10,11,12]),
                    'min_child_weight' : trial.suggest_int('min_child_weight',1,200),
                }
                xgb_reg_model = XGBClassifier(objective="binary:logistic",
                                        tree_method = 'gpu_hist',
                                        early_stopping_rounds=20,
                                        verbosity = 2, 
                                        eval_metric='logloss', 
                                        random_state=30,
                                        **param)
                xgb_reg_model.fit(X_train,y_train, eval_set = [(X_test,y_test)], verbose = 2)
                pred_xgb = xgb_reg_model.predict_proba(X_test)
                logloss = log_loss(y_test, pred_xgb, labels=[0,1], eps=1e-15)
                return logloss
            
            start = datetime.now()
            find_param = optuna.create_study(study_name="xgboost_exp", direction='minimize')
            find_param.optimize(objective, n_trials = 10)
            params = find_param.best_params
            print("Training Time = {}".format(datetime.now()-start))
            return params
        except Exception as e:
            raise ApplicationException(e, sys) from e
        
    def get_best_params_random_forest(self,X_train, y_train)->dict:
        try:
            logging.info("Running Hyperparameter Tuning for Random Forest Classifier")
            params = {"n_estimators" : [10, 20, 30, 50, 100, 200],
                      "max_depth" : [3,5,7,9,11,13,15],
                      "criterion": ["gini", "cross-entopy"],
                      "sampling_strategy": ["auto"]}
            rf_grid = GridSearchCV(RandomForestClassifier(random_state=30), 
                 param_grid=params, scoring="neg_log_loss", cv=5, refit=True, verbose=2)
            rf_grid.fit(X_train, y_train)
            best_params = rf_grid.best_params_
            return best_params
        except Exception as e:
            raise ApplicationException(e, sys) from e
        
    def train_xgboost(self, X_train, y_train, X_test, y_test):
        try:
            start = datetime.now()
            logging.info("Getting best parameters for XGBoost Classifier")
            params = self.get_best_params_xgboost(X_train, y_train, X_test, y_test)
            logging.info(f"Grid search completed. Best params : {params}")

            logging.info("Training XGBoost Classifier with best params")
            xgb_reg_model = XGBClassifier(objective="binary:logistic",
                                        tree_method = 'gpu_hist',
                                        verbosity = 1, 
                                        eval_metric='logloss', 
                                        random_state=30,
                                        **params)
            xgb_reg_model.fit(X_train,y_train, eval_set=[(X_test,y_test)],verbose = 1)
            logging.info("Training Time ={}".format(datetime.now()-start))
            return xgb_reg_model
        except Exception as e:
            raise ApplicationException(e, sys) from e
        
    def train_random_forest(self,X_train, y_train):
        try:
            start = datetime.now()
            logging.info("Getting best parameters for RandomForest Classifier")
            params = self.get_best_params_random_forest(X_train, y_train)
            
            logging.info(f"Grid search completed. Best params : {params}")
            logging.info("Training Random Forest Classifier with best params")
            rf= RandomForestClassifier(random_state=30, **params)
            rf.fit(X_train, y_train)
            logging.info("Training Time ={}".format(datetime.now()-start))
            return rf
        except Exception as e:
            raise ApplicationException(e, sys) from e
        
    def calibrated_model(self, model, X_train, y_train, method:str= "isotonic"):
        try:
            logging.info("Selected model calibration started")
            start = datetime.now()
            calib_model = CalibratedClassifierCV(base_estimator=model, method=method)
            calib_model.fit(X_train, y_train)
            logging.info("Model calibration completed successfully")
            print("Training Time =",datetime.now() - start)
            return calib_model
        except Exception as e:
            raise ApplicationException(e, sys) from e
        
    def best_model_selector(self, X_train, y_train, X_test, y_test, calibration:bool= True)->tuple:
        try:
            logging.info(f'{"*"*15} Training XGBoost Classifier {"*"*15}')
            xgb_model= self.train_xgboost(X_train, y_train, X_test, y_test)

            logging.info("Evaluating trained model.......")
            xgb_metrics= self.model_evaluation(xgb_model, "XGBoost", X_train, y_train, X_test, y_test)
            logging.info(f'{"*"*15} Training XGBoost Classifier Completed Successfully {"*"*15}')

            logging.info(f'{"*"*15} Training RandomForest Classifier {"*"*15}')
            rf_model= self.train_random_forest(X_train, y_train, X_test, y_test)

            logging.info("Evaluating trained model.......")
            rf_metrics= self.model_evaluation(rf_model, "RandomForest", X_train, y_train, X_test, y_test)
            logging.info(f'{"*"*15} Training RandomForest Classifier Completed Successfully {"*"*15}')

            logging.info("Selecting best model.....")
            if xgb_metrics[0] < rf_metrics[0]:  ## choosing model based on minimum log loss value
                model= xgb_model
                model_name= "XGBoost Classifier"
                logloss_value= xgb_metrics[1]
                roc_auc= xgb_metrics[2]
            else:
                model= rf_model
                model_name= "RandomForest Classifier"
                logloss_value= rf_metrics[1]
                roc_auc= rf_metrics[2]

            if calibration==True:
                model = self.calibrated_model(model, X_train, y_train, "isotonic")
                metrics = self.model_evaluation(model, f"Calibrated {model_name}", X_train, y_train, X_test, y_test)
                logloss_value= metrics[1]
                roc_auc= metrics[2]
            return (model, logloss_value, roc_auc)
        except Exception as e:
            raise ApplicationException(e, sys) from e
        
    def find_optimum_threshold(self, model, X_test, y_test)->float:
        try:
            logging.info("Finding optimum threshold.....")
            fpr, tpr, thresholds = roc_curve(y_test, model.predict_proba(X_test)[:,1])
            J = tpr - fpr
            ix = np.argmax(J)
            best_threshold = thresholds[ix]
            logging.info(f"Threshold found!!!.....Best Threshold = {best_threshold}")
            return float(best_threshold)
        except Exception as e:
            raise ApplicationException(e, sys) from e
        
    def model_evaluation(self, model, model_name, X_train, y_train, X_test, y_test)->tuple:
        try:
            logging.info(f"\n{'-'*15} {model_name} Performance {'-'*15}")
            predict_y  = model.predict_proba(X_train)
            log_loss_error = log_loss(y_train, predict_y, labels=[0,1], eps=1e-15)
            logging.info(f"{model_name} --> Train --> Log-Loss = {round(log_loss_error,4)}")
            train_roc= roc_auc_score(y_train, predict_y[:,1], labels=[0,1])
            logging.info(f"{model_name} --> Train --> ROC-AUC Score = {round(train_roc,4)}")

            predict_y  = model.predict_proba(X_test)
            log_loss_error = log_loss(y_test, predict_y, labels=[0,1], eps=1e-15)
            logging.info(f"{model_name} --> Test --> Log-Loss = {round(log_loss_error,4)}")
            test_roc= roc_auc_score(y_test, predict_y[:,1], labels=[0,1])
            logging.info(f"{model_name} --> Test --> ROC-AUC Score = {round(test_roc,4)}")

            self.plot_confusion_matrix(y_test, np.argmax(predict_y,axis=1))
            return (log_loss_error, test_roc)
        except Exception as e:
            raise ApplicationException(e, sys) from e

    
    def plot_confusion_matrix(self, model_name,test_y, predict_y)->None:
        # For Ploting confusion matrix
        C = confusion_matrix(test_y, predict_y)
        
        A = ((C.T)/(C.sum(axis=1))).T  # divide each element of confusion matrix with sum of elements in that column
        B = C/C.sum(axis=0) # divide each element of confusion matrix with sum of elements in that row
        
        plt.figure(figsize=(20,4))
        labels = [1,2]
        cmap = sns.light_palette("pink")
        
        plt.subplot(1,3,1)
        sns.heatmap(C, annot=True, cmap=cmap, fmt=".3f", xticklabels=labels, yticklabels=labels, annot_kws={"size": 15})
        plt.xlabel("Predicted Class",fontsize=15)
        plt.ylabel("Original Class",fontsize=15)
        plt.title("Confusion Matrix",fontsize=20)
        
        plt.subplot(1, 3, 2)
        sns.heatmap(B, annot=True, cmap=cmap, fmt=".3f", xticklabels=labels, yticklabels=labels, annot_kws={"size": 15})
        plt.xlabel('Predicted Class',fontsize=15)
        plt.ylabel('Original Class',fontsize=15)
        plt.title("Precision matrix",fontsize=20)
        
        plt.subplot(1, 3, 3)
        sns.heatmap(A, annot=True, cmap=cmap, fmt=".3f", xticklabels=labels, yticklabels=labels, annot_kws={"size": 15})
        plt.xlabel('Predicted Class',fontsize=15)
        plt.ylabel('Original Class',fontsize=15)
        plt.title("Recall matrix",fontsize=20)
        plt.suptitle(model_name, fontsize=20)
        
        plt.show()
        
    def initiate_model_training(self)-> ModelTrainerArtifact:
        try:
            logging.info("Finding Train and Test data paths")
            transformed_train_file_path = self.data_transformation_artifact.transformed_train_file_path
            transformed_test_file_path= self.data_transformation_artifact.transformed_test_file_path

            logging.info("Reading train and test dataframe")
            train_df= read_data(transformed_train_file_path)
            test_df= read_data(transformed_test_file_path)

            logging.info("Splitting train and test data into input and target feature")
            train_input_feature= train_df.drop(columns=["Id","is_duplicate"], axis=1)
            train_target_feature= train_df['is_duplicate']

            test_input_feature= test_df.drop(columns=["Id","is_duplicate"], axis=1)
            test_target_feature= test_df['is_duplicate']

            logging.info("Getting Trained Model.....")
            trained_model = self.best_model_selector(X_train=train_input_feature, y_train=train_target_feature,
                                                     X_test=test_input_feature, y_test=test_target_feature)
            
            # finding optimum threshold
            threshold = self.find_optimum_threshold(trained_model[0],test_input_feature,test_target_feature)

            logging.info("Saving Best model...")
            trained_model_file_path = self.model_trainer_config.trained_model_file_path
            create_directories([Path(os.path.dirname(trained_model_file_path))])
            save_bin(obj=trained_model[0], path=Path(trained_model_file_path))

            logloss_value= trained_model[1]
            roc_auc_score= trained_model[2]

            model_trainer_artifact = ModelTrainerArtifact(is_trained=True, message="Model Training Successfull",
                                                          trained_model_file_path=trained_model_file_path,
                                                          logloss_value=logloss_value,
                                                          roc_auc_score=roc_auc_score,
                                                          threshold=threshold)
            
            logging.info(f"Model Trainer Artifact: {model_trainer_artifact}")      
            return model_trainer_artifact
        except Exception as e:
            raise ApplicationException(e, sys) from e
        
    def __del__(self):
        logging.info(f"\n{'>'*20} Model Training Completed {'<'*20}\n\n")
