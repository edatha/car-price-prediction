import joblib
import uvicorn
import numpy as np
import pandas as pd
from pydantic import BaseModel

from fastapi import FastAPI

app = FastAPI(title='Car Price Prediction', version='1.0',
              description='Linear Regression model is used for prediction')

model = joblib.load("artifacts/best_model.joblib")

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
    """
     Home endpoint which can be used to test the availability of the application.
     """
    return {'message': 'Hallo Erlangga:)'}

# ML API endpoint for making prediction aganist the request received from client
@app.post("/predict")
def predict(data: Data):

    result = np.expm1(model.predict(pd.DataFrame(columns=['name','kms_driven','fuel_type','company','car_age'],data=np.array([data.name,data.kms_driven,data.fuel_type,data.company,data.car_age]).reshape(1,5)))[0])
    return result

if __name__ == '__main__':
    uvicorn.run("app:app", host="0.0.0.0", port=8000, reload=True)
