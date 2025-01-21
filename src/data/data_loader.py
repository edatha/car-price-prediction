import pandas as pd

from src.utils.config import config
from src.utils.logger import default_logger as logger

def load_data():
    """Loading data for preparring"""
    try:
        # Load data
        logger.info("Loading data from directory...")
        df = pd.read_csv(config.get('DATA_PATH'))
        logger.info("The data has been successfully loaded")
        
        return df
    
    except Exception as e:
        logger.error(f"Error in loading the data: {str(e)}")

