import pandas as pd
import numpy as np
import joblib
import yaml
import mlflow
from sksurv.ensemble import RandomSurvivalForest
from sksurv.metrics import concordance_index_censored

def main():
    # --- Загрузка параметров ---
    with open("params.yaml") as f:
        params = yaml.safe_load(f)
    n_est = params["train"]["n_estimators"]
    max_d = params["train"]["max_depth"]
    random_state = params["train"]["random_state"]

    # --- Загрузка данных ---
    X_train = pd.read_pickle("data/processed/X_train.pkl")
    X_test = pd.read_pickle("data/processed/X_test.pkl")
    y_train = np.load("data/processed/y_train.npy", allow_pickle=True)
    y_test = np.load("data/processed/y_test.npy", allow_pickle=True)

    # --- MLflow ---
    mlflow.set_experiment("RSF Survival")
    with mlflow.start_run():
        # --- Обучение ---
        rsf = RandomSurvivalForest(
            n_estimators=n_est,
            max_depth=max_d,
            random_state=random_state,
            n_jobs=-1
        )
        rsf.fit(X_train, y_train)

        # --- Оценка ---
        risk_scores = rsf.predict(X_test)
        c_index = concordance_index_censored(
            y_test['event'], y_test['time'], risk_scores
        )[0]

        # --- Логирование ---
        mlflow.log_param("n_estimators", n_est)
        mlflow.log_param("max_depth", max_d)
        mlflow.log_metric("c_index", c_index)
        mlflow.set_tag("model", "RandomSurvivalForest")

        # --- Сохранение модели ---
        joblib.dump(rsf, "model.pkl")
        mlflow.log_artifact("model.pkl")  # ← ключевая строка!

if __name__ == "__main__":
    main()