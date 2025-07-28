import joblib
from sklearn.metrics import mean_squared_error
from preprocess import load_data, preprocess, split_data, scale_data

def evaluate(file_path):
    df = load_data(file_path)
    df = preprocess(df)
    X_train, X_test, y_train, y_test = split_data(df)
    X_train_scaled, X_test_scaled, scaler = scale_data(X_train, X_test)

    model = joblib.load("model.pkl")
    predictions = model.predict(X_test_scaled)
    mse = mean_squared_error(y_test, predictions)
    print(f"Mean Squared Error: {mse:.2f}")

if __name__ == "__main__":
    evaluate("data/sales.csv")