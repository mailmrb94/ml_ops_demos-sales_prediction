import joblib
import numpy as np
from fastapi import FastAPI
from pydantic import BaseModel

app = FastAPI()
model = joblib.load("model.pkl")
scaler = joblib.load("scaler.pkl")

class SalesRequest(BaseModel):
    store: int
    item: int
    day: int
    month: int
    year: int
    day_of_week: int

@app.post("/predict")
def predict_sales(data: SalesRequest):
    input_data = np.array([[
        data.store,
        data.item,
        data.day,
        data.month,
        data.year,
        data.day_of_week
    ]])
    input_scaled = scaler.transform(input_data)
    prediction = model.predict(input_scaled)[0]
    return {"predicted_sales": prediction}