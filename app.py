import gradio as gr
import joblib
import numpy as np

model = joblib.load("model.pkl")
scaler = joblib.load("scaler.pkl")

def predict(store, item):
    input_data = np.array([[store, item]])
    input_scaled = scaler.transform(input_data)
    prediction = model.predict(input_scaled)[0]
    return round(prediction, 2)

demo = gr.Interface(
    fn=predict,
    inputs=[gr.Number(label="Store"), gr.Number(label="Item")],
    outputs=gr.Number(label="Predicted Sales"),
    title="Sales Forecasting App",
    description="Enter store and item ID to predict sales.",
)

demo.launch()
