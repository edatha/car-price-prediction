from src.data.data_loader import load_data
from src.data.data_preparation import data_preparation
from src.data.processing_pipeline import processingPipeline
from src.data.predict import predict, save_model
from src.utils.logger import default_logger as logger

if __name__ == "__main__":
    try:
        logger.info("LOADING DATA...")
        df = load_data()
        
        logger.info("PREPARING DATA...")
        X, y = data_preparation(df)
        
        logger.info("PIPELINE...")
        X_train, X_test, y_train, y_test, pipe = processingPipeline(X, y)
        
        logger.info("PREDICTING...")
        predict(X_train, X_test, y_train, y_test, pipe)
        
        logger.info("SAVING...")
        save_model(pipe)
        
    except Exception as e:
        logger.error(f"Error: {str(e)}")