import os
import joblib
import numpy as np
import pandas as pd

from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error

from src.utils.config import config
from src.utils.logger import default_logger as logger

def predict(X_train, X_test, y_train, y_test, pipe):
    """Predicting data"""
    try:
        # Use Linear Regression
        # logger.info("Predicting using LinearRegression")
        # lr = LinearRegression()
        # pipe = make_pipeline(column_trans, lr)
        
        # Log the shapes of X_train and y_train
        logger.info(f"Shape of X_train: {X_train.shape}")
        logger.info(f"Shape of y_train: {y_train.shape}")
        y_train = np.array(y_train).reshape(-1,1)
        
        logger.info(f"Shape of X_train: {X_train.shape}")
        logger.info(f"Shape of y_train: {y_train.shape}")
        # Fit the model
        pipe.fit(X_train, y_train)
        
        logger.info("Predicting testing data")
        y_pred = pipe.predict(X_test)
        
        # Revert log transformation for evaluation
        # y_test_orig = np.expm1(y_test)
        # y_pred_orig = np.expm1(y_pred)
        
        logger.info(f"R2 Test -> {r2_score(y_test, y_pred)}")
        logger.info(f"MAE Test -> {mean_absolute_error(y_test, y_pred)}")
        logger.info(f"RMSE Test -> {np.sqrt(mean_squared_error(y_test, y_pred))}")
    
    except Exception as e:
        logger.error(f"Error in predicting data {e}")
        raise
    
def save_model(pipe):
    try:
        logger.info("Saving the model...")
        model_path = config.get('MODEL_PATH')
        joblib.dump(pipe, open(model_path, 'wb'))

        logger.info(f"Model saved to {model_path}")
        
    except Exception as e:
        logger.error(f"Error in saving the model {e}")
        raise