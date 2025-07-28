import joblib
import mlflow
import numpy as np
from mlflow.models.signature import infer_signature
from sklearn.ensemble import RandomForestRegressor
from preprocess import load_data, preprocess, split_data, scale_data

mlflow.set_experiment("demand_forecast")

def train(file_path):
    df = load_data(file_path)
    df = preprocess(df)
    X_train, X_test, y_train, y_test = split_data(df)
    X_train_scaled, X_test_scaled, scaler = scale_data(X_train, X_test)

    model = RandomForestRegressor(n_estimators=100, random_state=42)
    with mlflow.start_run():
        model.fit(X_train_scaled, y_train)

        sample_input = np.array([X_train_scaled[0]])
        sample_output = model.predict(sample_input)
        signature = infer_signature(sample_input, sample_output)

        mlflow.sklearn.log_model(
            sk_model=model,
            name="model",
            signature=signature,
            input_example=sample_input
        )

        mlflow.log_param("n_estimators", 100)

        joblib.dump(model, "model.pkl")
        joblib.dump(scaler, "scaler.pkl")

if __name__ == "__main__":
    train("data/sales.csv")
