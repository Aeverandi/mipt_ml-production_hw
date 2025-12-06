import pandas as pd
import numpy as np
import yaml
import mlflow
import mlflow.xgboost
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import xgboost as xgb

def main():
    # --- Загрузка параметров ---
    with open("params.yaml") as f:
        params = yaml.safe_load(f)
    model_params = params["train"]["model_params"]
    random_state = params["prepare"]["random_state"]
    # --- Загрузка данных ---
    X_train = pd.read_pickle("data/processed/X_train.pkl")
    X_test = pd.read_pickle("data/processed/X_test.pkl")
    y_train = np.load("data/processed/y_train.npy", allow_pickle=True)
    y_test = np.load("data/processed/y_test.npy", allow_pickle=True)

    # --- MLflow ---
    mlflow.set_tracking_uri("sqlite:///mlflow.db")
    mlflow.set_experiment("ef_prediction")
    mlflow.xgboost.autolog(log_models=False)

    with mlflow.start_run():
        # Логируем параметры вручную
        mlflow.log_params({
            "test_size": params["prepare"]["test_size"],
            **model_params
        })

        # --- Обучение ---
        model = xgb.XGBRegressor(
            random_state=random_state,
            **model_params
        )
        model.fit(X_train, y_train)

        # --- Прогноз и метрики ---
        y_pred = model.predict(X_test)

        mae = mean_absolute_error(y_test, y_pred)
        rmse = np.sqrt(mean_squared_error(y_test, y_pred))
        r2 = r2_score(y_test, y_pred)

        # Логируем все три метрики
        mlflow.log_metrics({
            "MAE": mae,
            "RMSE": rmse,
            "R2": r2
        })

        # --- Сохранение модели как артефакта ---
        model.save_model("model.json")
        mlflow.log_artifact("model.json")

        print(f"MAE: {mae:.4f}, RMSE: {rmse:.4f}, R2: {r2:.4f}")

if __name__ == "__main__":
    main()