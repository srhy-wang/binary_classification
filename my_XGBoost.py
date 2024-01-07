import xgboost as xgb
from xgboost import plot_importance
import sklearn.metrics as skm
from sklearn.model_selection import GridSearchCV
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

class Myxgboost:
    def __init__(self, objective='binary:hinge', verbosity=0, 
                        booster='gbtree', tree_method='auto', n_jobs=1,
                        max_delta_step=0, subsample=1, 
                        colsample_bytree=1, colsample_bylevel=1, colsample_bynode=1, 
                        scale_pos_weight=1, base_score=0.5, random_state=0, metrics='accuracy'):
        
        self.n_estimators = None # 迭代次数 基学习器数量
        self.max_depth = None # 最大深度
        self.min_child_weight = None # 最小子权重
        self.gamma = None # 最小损失
        self.reg_alpha = None
        self.reg_lambda = None
        self.learning_rate = None

        self.objective = objective
        self.verbosity = verbosity
        self.booster = booster
        self.tree_method = tree_method
        self.n_jobs = n_jobs
        self.subsample = subsample
        self.colsample_bytree = colsample_bytree
        self.colsample_bylevel = colsample_bylevel
        self.colsample_bynode = colsample_bynode
        self.scale_pos_weight = scale_pos_weight
        self.base_score = base_score
        self.random_state = random_state
        self.max_delta_step = max_delta_step

        self.model = xgb.XGBClassifier

        if metrics.lower() == 'accuracy': # 精确度
            self.metrics = skm.accuracy_score
        elif metrics.lower() == 'logloss': # 对数损失
            self.metrics = skm.log_loss
        elif metrics.lower() == 'precision': # 查准率 精度
            self.metrics = skm.precision_score
        elif metrics.lower() == 'f1score': # f1值
            self.metrics = skm.f1_score
        elif metrics.lower() == 'recall': # 查全率
            self.metrics = skm.recall_score
        # elif metrics.lower() == 'auc':
        #     self.metrics == skm.roc_auc_score
        else:
            raise Exception("please choose metrics from ['accuracy', 'precision', 'f1score', 'logloss', 'recall']")
        
    def plot_param(self, gsearch, param):
        scores = gsearch.cv_results_['mean_test_score']
        params = gsearch.cv_results_['params']
        parameters_score = pd.DataFrame(params, scores)
        parameters_score['means_score'] = parameters_score.index
        parameters_score = parameters_score.reset_index(drop=True)
        plt.figure(figsize=(15, 12))
        plt.subplot(2, 1, 1)
        plt.plot(parameters_score.iloc[:, :-1], 'o-')
        plt.legend(parameters_score.columns.to_list()[:-1], loc='upper left')
        plt.title('Parameters', loc='left', fontsize='xx-large', fontweight='heavy')
        plt.subplot(2, 1, 2)
        plt.plot(parameters_score.iloc[:, -1], 'r+-')
        plt.legend(parameters_score.columns.to_list()[-1:], loc='upper left')
        plt.title('Scores', loc='left', fontsize='xx-large', fontweight='heavy')
        plt.savefig(f'xgb/{param}_xgb_param.jpg')
        # plt.show()


# 1 对n_estimators_opt进行网格搜索
    def n_estimators_opt(self, x, y, n_estimators_param):
        param = {'n_estimators': n_estimators_param}
        gsearch = GridSearchCV(estimator=xgb.XGBClassifier(
            max_depth=8, min_child_weight=1, gamma=0, 
            reg_alpha=0, reg_lambda=1, learning_rate=0.1,
            random_state=10
        ), param_grid=param, scoring='accuracy', cv=5)  #scoring='roc_auc'
        gsearch.fit(x, y)
        # return gsearch.grid_scores_, gsearch.best_params_, gsearch.best_score_
        self.n_estimators = gsearch.best_params_['n_estimators']
        print(f'n_estimators:{self.n_estimators}, best_score:{gsearch.best_score_}')
        self.plot_param(gsearch, 'n_estimators')

# 2 对max_depth和min_child_weight进行网格搜索
    def max_depth_min_child_opt(self, x, y, max_depth_param, min_child_weight_param):
        param = {'max_depth':max_depth_param, 'min_child_weight':min_child_weight_param}
        gsearch = GridSearchCV(estimator=xgb.XGBClassifier(
            n_estimators=self.n_estimators, gamma=0, 
            reg_alpha=0, reg_lambda=1, learning_rate=0.1, 
            random_state=10
        ), param_grid=param, scoring='accuracy', cv=5)
        gsearch.fit(x, y)
        self.max_depth = gsearch.best_params_['max_depth']
        self.min_child_weight = gsearch.best_params_['min_child_weight']
        print(f'max_depth:{self.max_depth}, min_child_weight:{self.min_child_weight}, best_score:{gsearch.best_score_}')
        self.plot_param(gsearch, 'max_depth_min_child')

# 3 对gamma进行网格搜索
    def gamma_opt(self, x, y, gamma_param):
        param = {'gamma':gamma_param}
        gsearch = GridSearchCV(estimator=xgb.XGBClassifier(
            n_estimators=self.n_estimators, max_depth=self.max_depth, min_child_weight=self.min_child_weight, 
            reg_alpha=0, reg_lambda=1, learning_rate=0.1, 
            random_state=10
        ), param_grid=param, scoring='accuracy', cv=5)
        gsearch.fit(x, y)
        self.gamma = gsearch.best_params_['gamma']
        print(f'gamma:{self.gamma}, best_score:{gsearch.best_score_}')
        self.plot_param(gsearch, 'gamma')

# 4 对reg_alpha、reg_lambda进行网格搜索
    def reg_alpha_lambda_opt(self, x, y, reg_alpha_param, reg_lambda_param):
        param = {'reg_alpha':reg_alpha_param, 'reg_lambda':reg_lambda_param}
        gsearch = GridSearchCV(estimator=xgb.XGBClassifier(
            n_estimators=self.n_estimators, max_depth=self.max_depth, min_child_weight=self.min_child_weight, 
            gamma=self.gamma, learning_rate=0.1, random_state=10
        ), param_grid=param, scoring='accuracy', cv=5)
        gsearch.fit(x, y)
        self.reg_alpha = gsearch.best_params_['reg_alpha']
        self.reg_lambda = gsearch.best_params_['reg_lambda']
        print(f'reg_alpha:{self.reg_alpha}, reg_lambda:{self.reg_lambda}, best_score:{gsearch.best_score_}')
        self.plot_param(gsearch, 'reg_alpha_lambda')

# 5 对learning_rate进行网格搜索
    def learning_rate_opt(self, x, y, learning_rate_param):
        param = {'learning_rate':learning_rate_param}
        gsearch = GridSearchCV(estimator=xgb.XGBClassifier(
            n_estimators=self.n_estimators, max_depth=self.max_depth, min_child_weight=self.min_child_weight, 
            reg_alpha=self.reg_alpha, reg_lambda=self.reg_lambda,
            random_state=10
        ), param_grid=param, scoring='accuracy', cv=5)
        gsearch.fit(x, y)
        self.learning_rate= gsearch.best_params_['learning_rate']
        print(f'learning_rate:{self.learning_rate}, best_score:{gsearch.best_score_}')
        self.plot_param(gsearch, 'learning_rate')

# 6 对特征数量进行网格搜索
    # def colsample_bytree_opt(self, x, y, colsample_bytree_param):
    #     param = {'colsample_bytree':colsample_bytree_param}
    #     gsearch = GridSearchCV(estimator=xgb.XGBClassifier(
    #         n_estimators=self.n_estimators, max_depth=self.max_depth, min_child_weight=self.min_child_weight, 
    #         reg_alpha=self.reg_alpha, reg_lambda=self.reg_lambda, learning_rate=self.learning_rate,
    #         random_state=10
    #     ), param_grid=param, scoring='accuracy', cv=5)
    #     gsearch.fit(x, y)
    #     self.colsample_bytree= gsearch.best_params_['colsample_bytree']
    #     print(f'colsample_bytree:{self.colsample_bytree}, best_score:{gsearch.best_score_}')


    def fit(self, x_train, y_train, n_estimators=50, max_depth=15, min_child_weight=1, 
            gamma=0, reg_alpha=0, reg_lambda=1, learning_rate=0.1):
        self.n_estimators_opt(x_train, y_train, n_estimators) # 得到n_estimator
        self.max_depth_min_child_opt(x_train, y_train, max_depth, min_child_weight) # 得到max_depth和min_child_weight
        self.gamma_opt(x_train, y_train, gamma) # 得到gamma
        self.reg_alpha_lambda_opt(x_train, y_train, reg_alpha, reg_lambda) # 得到reg_alpha和reg_lambda
        self.learning_rate_opt(x_train, y_train, learning_rate) # 得到learning_rate
        self.model = self.model(max_depth=self.max_depth, learning_rate=self.learning_rate, object=self.objective,
                                n_estimators=self.n_estimators, verbosity=self.verbosity, booster=self.booster, 
                                tree_method=self.tree_method, n_jobs=self.n_jobs,
                                gamma=self.gamma, min_child_weight=self.min_child_weight, max_delta_step=self.max_delta_step, 
                                subsample=self.subsample, colsample_bytree=self.colsample_bytree,
                                colsample_bylevel=self.colsample_bylevel, colsample_bynode=self.colsample_bynode, 
                                reg_alpha=self.reg_alpha, reg_lambda=self.reg_lambda,
                                scale_pos_weight=self.scale_pos_weight, base_score=self.base_score, 
                                random_state=self.random_state
                                )
        self.model.fit(x_train, y_train)
        self.classes = self.model.classes_
        self.feature_importances = self.model.get_booster().get_fscore()
        self.feature_importances = sorted(self.feature_importances.items(), key=lambda x: x[1], reverse=True)
        self.n_features_in = self.model.n_features_in_

    def predict(self, x):
        y_pred = self.model.predict(x)
        return y_pred
    
    def pred_acc(self, x, y):
        y_pred = self.predict(x)
        metric = self.metrics(y, y_pred)
        return f'{self.metrics.__name__}:{metric}'
    
    def plot_importance(self):
        plot_importance(self.model)
        plt.savefig('xgb/xgb_importance.jpg')
        # plt.show()
