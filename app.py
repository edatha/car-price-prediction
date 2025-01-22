import joblib
import uvicorn
import numpy as np
import pandas as pd
from pydantic import BaseModel

from fastapi import FastAPI

app = FastAPI(
    title='Car Price Prediction',
    version='1.0',
    description='Linear Regression model is used for prediction'
)

# Load the model
import os
model_path = os.path.join(os.path.dirname(__file__), "artifacts/best_model.joblib")
model = joblib.load(model_path)

class Data(BaseModel):
    name: str
    kms_driven: int 
    fuel_type: str
    company: str
    car_age: int

# API root
@app.get('/')
@app.get('/home')
def read_home():
    return {'message': 'Hallo Erlangga :)'}

# ML API endpoint for prediction
@app.post("/predict")
def predict(data: Data):
    # Logging input data
    print(f"Input Data: {data}")
    
    try:
        result = np.expm1(
            model.predict(
                pd.DataFrame(
                    columns=['name', 'kms_driven', 'fuel_type', 'company', 'car_age'],
                    data=np.array([data.name, data.kms_driven, data.fuel_type, data.company, data.car_age]).reshape(1, 5)
                )
            )[0]
        )
        return {"predicted_price": result}
    except Exception as e:
        return {"error": str(e)}

if __name__ == '__main__':
    host = os.getenv("API_HOST", "0.0.0.0")
    port = int(os.getenv("API_PORT", 8000))
    uvicorn.run("app:app", host=host, port=port, reload=True)
