# Домашнее задание №1
Выполнил: **Лобан Константин Михайлович, группа М08-402ПА**

Источник датасета: https://www.kaggle.com/datasets/aadarshvelu/heart-failure-prediction-clinical-records

Предназначение модели: прогнозирование летальности для пациентов с сердечной недостаточностью
## Инструкция по локальному развертыванию
```
git clone https://github.com/Aeverandi/mipt_ml-production_hw.git
cd mipt_ml-production_hw
pip install -r requirements.txt
dvc pull
dvc repro
```
## Примечания автора
У меня почему-то так и не получилось воспроизвести логирование обучения модели в ML-flow. На stackowerflow кто-то писал, что модели из библиотеки scikit-survival не логируются автоматически, но с другой стороны я задавал логирование вручную, как Вы писали в тексте ДЗ. В общем, многочисленные попытки залогировать обучение модели не принесли результата, я уже готов потерять эти 2 балла. Выбор и тестирование модели проводилось в ноубуке [google-colab](https://colab.research.google.com/drive/1fETA9j2hP9gxz5i4NR-ZE0alAcQdibR2?usp=sharing).

