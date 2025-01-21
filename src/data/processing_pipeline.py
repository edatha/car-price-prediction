import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.linear_model import LinearRegression

from src.utils.config import config
from src.utils.logger import default_logger as logger

def processingPipeline(X: pd.DataFrame, y: pd.Series):
    """Create a pipeline for processing data

    Args:
        X (pd.DataFrame): _description_
        y (pd.Series): _description_
    """
    try:
        logger.info("Creating a pipeline for processing...")
        ohe = OneHotEncoder(handle_unknown='ignore')
        scaler = StandardScaler()
        
        categorical_cols = config.get('CATEGORICAL_COLS')
        numeric_cols = config.get('NUMERICAL_COL')
        
        column_trans = ColumnTransformer(transformers=[
        ('ohe', ohe, categorical_cols),  # Apply OneHotEncoder to categorical columns
        ('scaler', scaler, numeric_cols)  # Apply StandardScaler to numeric columns
        ])
        
        pipe = Pipeline(steps=[
        ('preprocessor', column_trans),  # Preprocessing step
        ('model', LinearRegression())   # Model step
        ])
        
        # Split our data into X_train, X_test, y_train, y_test
        logger.info("Splitting our data into X_train, X_test, y_train, y_test")
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=config.get('TEST_SIZE'), random_state=config.get('RANDOM_STATE'))
        
        return X_train, X_test, y_train, y_test, pipe
    
    except Exception as e:
        logger.error(f"Error in creating pipeline {e}")
        raise