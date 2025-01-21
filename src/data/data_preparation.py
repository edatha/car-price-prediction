import pandas as pd
import numpy as np

from src.utils.config import config
from src.utils.logger import default_logger as logger

def data_preparation(df: pd.DataFrame, target: str = config.get('TARGET_COLUMN')):
    """Preparing data for modelling"""
    try:
        # Remove outliers using IQR
        logger.info("Removing outliers from the data")
        Q1 = df[target].quantile(0.25)
        Q3 = df[target].quantile(0.75)
        IQR = Q3 - Q1
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR
        df = df[(df[target] >= lower_bound) & (df[target] <= upper_bound)].copy()
        
        # Log transform target to handle skewness
        # df[target] = np.log1p(df[target])
        
        # Create new features
        logger.info(f"Creating a new feature: {config.get('NEW_COLUMN')}")
        df.loc[:, config.get('NEW_COLUMN')] = 2025 - df[config.get('DROP_COLUMN_1')]
        
        # Drop unnessary columns
        df = df.drop(columns=config.get('DROP_COLUMNS'), axis=1)
        
        # Split our data into X and y
        logger.info("Splitting our data into X and y")
        X = df.drop(columns=[target], axis=1)
        y = df[target]
        
        logger.info("Preparing data has been completed")
        return X, y
    
    except Exception as e:
        logger.error(f"Error in preparing data: {str(e)}")
        raise

        
    