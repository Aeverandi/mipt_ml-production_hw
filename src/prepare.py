import pandas as pd
import numpy as np
import os
import yaml
from sklearn.model_selection import train_test_split

def main():
    with open("params.yaml") as f:
        params = yaml.safe_load(f)

    test_size = params["prepare"]["test_size"]
    random_state = params["prepare"]["random_state"]

    # Датасет с каггл: https://www.kaggle.com/datasets/aadarshvelu/heart-failure-prediction-clinical-records
    df = pd.read_csv("data/raw/heart_failure_clinical_records.csv")

    y = df['ejection_fraction']
    X = df.drop(columns=['DEATH_EVENT', 'time', 'ejection_fraction'])

    X_train, X_test, y_train, y_test = train_test_split(X, y,
            test_size=test_size,
            random_state=random_state)

    # Сохраняем
    os.makedirs("data/processed", exist_ok=True)
    X_train.to_pickle("data/processed/X_train.pkl")
    X_test.to_pickle("data/processed/X_test.pkl")
    np.save("data/processed/y_train.npy", y_train)
    np.save("data/processed/y_test.npy", y_test)

if __name__ == "__main__":
    main()