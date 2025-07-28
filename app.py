print("Launching Gradio app...")

import gradio as gr
import joblib
import numpy as np

model = joblib.load("model.pkl")
scaler = joblib.load("scaler.pkl")

def predict_sales(store, item, day, month, year, day_of_week):
    input_data = np.array([[store, item, day, month, year, day_of_week]])
    input_scaled = scaler.transform(input_data)
    prediction = model.predict(input_scaled)[0]
    return round(prediction, 2)

iface = gr.Interface(
    fn=predict_sales,
    inputs=[
        gr.Number(label="Store ID"),
        gr.Number(label="Item ID"),
        gr.Number(label="Day"),
        gr.Number(label="Month"),
        gr.Number(label="Year"),
        gr.Number(label="Day of Week")
    ],
    outputs=gr.Textbox(label="Predicted Sales"),
    title="Sales Forecasting UI",
    description="Enter the date and product details to get a sales prediction."
)

print("Ready to launch on browser...")
iface.launch()
