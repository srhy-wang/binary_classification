## File description:
three binary classification for data (https://www.kaggle.com/datasets/kukuroo3/body-signal-of-smoking) 

- plot_code.ipynb: there is some descriptions of the dataset
- preprocess.py: there is a funtion to proprecess raw_dataset


- my_lr.py: there is a LogisticRegression model base on sklearn.linear_model.LogisticRegressionCV
- my_randoforest.py: there is a RandomForest model base on sklearn.ensemble.RandomForestClassifier
- my_XGBoost.py: there is a XGBoost model base on xgboost.XGBClassifier
  - All of the three class include methods for fitting, tuning, predicting, and evaluating models
    
- main.py: there are the main functions of the project
  
- lr folder: LogisticRegression results
- rf folder: RandomForest results
- xgb folder: XGBoost results
  - some figures
## Get started from here!

#### Requirements
1. Python 3.11.4
2. conda 23.7.4
3. xgboost 2.0.3
4. pandas 1.5.3
5. numpy 1.24.3
6. matplotlib 3.8.0
7. sklearn 1.3.0
8. seaborn 0.12.2
#### Run    
python -u main.py
