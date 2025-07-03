from fastapi import FastAPI
from pydantic import BaseModel
import joblib
import pandas as pd
import numpy as np
from typing import Dict

app = FastAPI(title="Loan Default Prediction API", version="1.0.0")

# Load model and preprocessors
model = joblib.load("data/models/best_model_XGBoost.joblib")

class LoanApplication(BaseModel):
    Client_Income: float
    Car_Owned: int
    Bike_Owned: int
    Active_Loan: int
    House_Own: int
    Child_Count: int
    Credit_Amount: float
    Loan_Annuity: float
    Age_Days: int
    Employed_Days: int
    Client_Family_Members: int
    # Add other required features

class PredictionResponse(BaseModel):
    prediction: int
    probability: float
    risk_level: str

@app.post("/predict", response_model=PredictionResponse)
async def predict_default(loan_data: LoanApplication):
    # Convert to DataFrame
    df = pd.DataFrame([loan_data.dict()])
    
    # Feature engineering (same as training)
    df['Age_Years'] = abs(df['Age_Days']) / 365.25
    df['Is_Employed'] = (df['Employed_Days'] > 0).astype(int)
    df['Debt_to_Income'] = df['Credit_Amount'] / (df['Client_Income'] + 1)
    # Add other engineered features...
    
    # Make prediction
    prediction = model.predict(df)[0]
    probability = model.predict_proba(df)[0][1]
    
    # Risk level
    if probability >= 0.7:
        risk_level = "High Risk"
    elif probability >= 0.3:
        risk_level = "Medium Risk"
    else:
        risk_level = "Low Risk"
    
    return PredictionResponse(
        prediction=int(prediction),
        probability=float(probability),
        risk_level=risk_level
    )

@app.get("/health")
async def health_check():
    return {"status": "healthy"}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)