from fastapi import FastAPI
from pydantic import BaseModel
import joblib
import pandas as pd


app = FastAPI()

model = joblib.load('xgb_model.pkl')
model_columns = joblib.load('model_columns.pkl')


class PatientData(BaseModel):
    data: dict

@app.post("/predict")
def predict(patient: PatientData):
    
    df = pd.DataFrame([patient.data])
    
    
    df = pd.get_dummies(df)
    
    
    df = df.reindex(columns=model_columns, fill_value=0)
    
    
    prediction = model.predict(df)
    
    
    return {"prediction": int(prediction[0])}