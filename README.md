# Домашнее задание №1
Выполнил: **Лобан Константин Михайлович, группа М08-402ПА**

Источник датасета: https://www.kaggle.com/datasets/aadarshvelu/heart-failure-prediction-clinical-records

Предназначение модели: прогнозирование летальности для пациентов с сердечной недостаточностью
## Инструкция по локальному развертыванию
```
git clone --branch new_ef_predict --single-branch https://github.com/Aeverandi/mipt_ml-production_hw.git
cd mipt_ml-production_hw
pip install -r requirements.txt
dvc repro
mlflow ui --backend-store-uri sqlite:///mlflow.db
```
## Примечания автора
Несколько дней промучался, так и не получилось запушить изменения в новой ветке через терминал. Как наверное понятно из коммитов, сделал это вручную. Удалили все локально, проверил с нуля воспроизводимость - вроде работает.  

