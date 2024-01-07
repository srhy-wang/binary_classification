import pandas as pd
import numpy as np
from my_lr import Mylogisticregression
from my_randomforest import Myrandomforest
from my_XGBoost import Myxgboost
from preprocess import pretransX, prescalarX, preprocessY

x_test = pd.read_csv('competition_format/x_test.csv', index_col='ID')
x_train = pd.read_csv('competition_format/x_train.csv', index_col='ID')
y_test = pd.read_csv('competition_format/y_test.csv', index_col='ID')
y_train = pd.read_csv('competition_format/y_train.csv', index_col='ID')

if __name__ == '__main__':
# preprocess
    x_train = pretransX(x_train)
    x_test = pretransX(x_test)
    x_train_scaled = prescalarX(x_train)
    x_test_scaled = prescalarX(x_test)
    y_train = preprocessY(y_train, x_train)
    y_test = preprocessY(y_test, x_test)

    x_train_save = pd.DataFrame(x_train_scaled, columns=x_train.columns)
    x_train_save.to_csv('process_data/x_train_processed.csv')
    x_test_save = pd.DataFrame(x_test_scaled, columns=x_test.columns)
    x_test_save.to_csv('process_data/x_test_processed.csv')

    
# LR
    cs_params = range(5,16)
    lr = Mylogisticregression(cv=10, metrics='accuracy')
    lr.fit(x_train_scaled, y_train, cs_params)
    lr_train_error = lr.pred_acc(x_train_scaled, y_train)
    lr_test_error = lr.pred_acc(x_test_scaled, y_test)
    coef = pd.DataFrame(lr.coef, columns=x_train.columns)
    print('logisticregression results:')
    print(coef.T)
    print(f'lr_train_error:{lr_train_error}')
    print(f'lr_test_error:{lr_test_error}')
# RF
    n_estimators_param = range(30,101,10)
    max_depth_param = range(15,51,5)
    min_samples_split_param = range(80,150,20)
    min_samples_leaf_param = range(10,60,10)
    max_features_param = range(5,25,2)

    rf = Myrandomforest(# n_estimators_param=n_estimators_param, max_depth_param=max_depth_param, 
                        # min_samples_split_param=min_samples_split_param, min_samples_leaf_param=min_samples_leaf_param,
                        #  max_features_param=max_features_param, 
                         metrics='accuracy')
    rf.fit(x_train_scaled, y_train, n_estimators_param, max_depth_param, 
           min_samples_split_param, min_samples_leaf_param, max_features_param)
    x = pd.DataFrame(x_train_scaled, columns=x_train.columns)
    rf.plot_importance(x)
    rf_train_error = rf.pred_acc(x_train_scaled, y_train)
    rf_test_error = rf.pred_acc(x_test_scaled, y_test)
    print(f'rf_train_error:{rf_train_error}')
    print(f'rf_test_error:{rf_test_error}')
# XGB
     # 迭代次数 基学习器数量
     # 最大深度
    min_child_weight_param = range(1,11,2) # 最小子权重
    gamma_param = [0.05, 0.1, 0.2, 0.3] # 最小损失
    reg_alpha_param = [0.01, 0.02, 0.03, 0.04, 0.05, 0.1, 0.2]
    reg_lambda_param = [0.05, 0.1, 0.5, 1, 2, 3]
    learning_rate_param = [0.01, 0.02, 0.03, 0.05, 0.1, 0.2, 0.3]
    xgb = Myxgboost( metrics='accuracy')
    xgb.fit(x_train_scaled, y_train, n_estimators=n_estimators_param, max_depth=max_depth_param,
            min_child_weight=min_child_weight_param, gamma=gamma_param, reg_alpha=reg_alpha_param,
                    reg_lambda=reg_lambda_param, learning_rate=learning_rate_param)
    xgb.plot_importance()
    xgb_train_error = xgb.pred_acc(x_train_scaled, y_train)
    xgb_test_error = xgb.pred_acc(x_test_scaled, y_test)
    print(f'xgb_train_error:{xgb_train_error}')
    print(f'xgb_test_error:{xgb_test_error}')