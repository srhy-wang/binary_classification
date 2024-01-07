import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV
import sklearn.metrics as skm

class Myrandomforest:
    def __init__(self, criterion='gini', min_impurity_decrease=0, #max_features='sqrt', bootstrap=True,
                    oob_score=True, max_leaf_nodes = None, # n_jobs=None, 
                    random_state=0, # verbose=0, warm_start=False, class_weight=None, ccp_alpha=0, max_samples=None,
                    # n_estimators_param=100, max_depth_param=None, min_samples_split_param=2,
                    # min_samples_leaf_param=1, max_features_param='sqrt', 
                    metrics='accuracy'):
        self.n_estimators = None # 基学习器数量*
        self.criterion = criterion
        self.max_depth = None # 最大深度*
        self.min_samples_split = None # 内部节点再划分所需最小样本数*
        self.max_features = None # 最大特征数*
        self.min_samples_leaf = None # 叶子节点最少样本数*
        self.max_leaf_nodes = max_leaf_nodes # 最大叶子节点数
        self.min_impurity_decrease = min_impurity_decrease # 节点划分最小不纯度
        self.oob_score__ = oob_score
        self.random_state = random_state

        self.model = RandomForestClassifier
        self.base_estimators = None # 基学习器
        self.n_features_in = None # 特征数
        self.feature_importances = None # 特征重要性
        self.oob_score = None # oob得分

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
        plt.savefig(f'rf/{param}_rf_param.jpg')
        # plt.show()


# 1 对n_estimators_opt进行网格搜索
    def __n_estimators_opt(self, x, y, n_estimators_param):
        param = {'n_estimators': n_estimators_param}
        gsearch = GridSearchCV(estimator=RandomForestClassifier(
            min_samples_split=100, min_samples_leaf=20, max_depth=8, max_features='sqrt' # ,oob_score=True
            , random_state=10
        ), param_grid=param, scoring='accuracy', cv=5)  #scoring='roc_auc'
        gsearch.fit(x, y)
        # return gsearch.grid_scores_, gsearch.best_params_, gsearch.best_score_
        self.n_estimators = gsearch.best_params_['n_estimators']
        print(f'n_estimators:{self.n_estimators}, best_score:{gsearch.best_score_}')
        self.plot_param(gsearch, 'n_estimators')

# 2 对max_depth和min_samples_split进行网格搜索
    def __max_depth_min_samples_opt(self, x, y, max_depth_param, min_samples_split_param):
        param = {'max_depth':max_depth_param, 'min_samples_split':min_samples_split_param}
        gsearch = GridSearchCV(estimator=RandomForestClassifier(
            n_estimators=self.n_estimators, min_samples_leaf=20, random_state=10 #, oob_score=True
        ), param_grid=param, scoring='accuracy', cv=5)
        gsearch.fit(x, y)
        self.max_depth = gsearch.best_params_['max_depth']
        # self.min_samples_split = gsearch.best_params_['min_samples_opt']
        print(f'max_depth:{self.max_depth}, best_score:{gsearch.best_score_}')
        self.plot_param(gsearch, 'max_depth_min_samples')

# 3 对min_samples_leaf和min_samples_split进行网格搜索
    def __min_samples_leaf_split_opt(self, x, y, min_samples_split_param, min_samples_leaf_param):
        param = {'min_samples_split':min_samples_split_param, 'min_samples_leaf':min_samples_leaf_param}
        gsearch = GridSearchCV(estimator=RandomForestClassifier(
            n_estimators=self.n_estimators, max_depth=self.max_depth, random_state=10 #, oob_score=True
        ), param_grid=param, scoring='accuracy', cv=5)
        gsearch.fit(x, y)
        self.min_samples_split = gsearch.best_params_['min_samples_split']
        self.min_samples_leaf = gsearch.best_params_['min_samples_leaf']
        print(f'min_samples_split:{self.min_samples_split}, min_samples_leaf:{self.min_samples_leaf}, best_score:{gsearch.best_score_}')
        self.plot_param(gsearch, 'min_samples_leaf_split')

# 4 对max_features进行网格搜索
    def __max_features_opt(self, x, y, max_features_param):
        param = {'max_features':max_features_param}
        gsearch = GridSearchCV(estimator=RandomForestClassifier(
            n_estimators=self.n_estimators, min_samples_split=self.min_samples_split, 
            min_samples_leaf=self.min_samples_leaf, max_depth=self.max_depth, 
            oob_score=True, random_state=10
        ), param_grid=param, scoring='accuracy', cv=5)  #scoring='roc_auc'
        gsearch.fit(x, y)
        self.max_features = gsearch.best_params_['max_features']
        print(f'max_features:{self.max_features}, best_score:{gsearch.best_score_}')
        self.plot_param(gsearch, 'max_features')


    def fit(self, x_train, y_train, n_estimators_param=100, max_depth_param=None, min_samples_split_param=2,
                     min_samples_leaf_param=1, max_features_param='sqrt'):
        self.__n_estimators_opt(x_train, y_train, n_estimators_param) # 得到n_estimator
        self.__max_depth_min_samples_opt(x_train, y_train, max_depth_param, min_samples_split_param) # 得到max_depth
        self.__min_samples_leaf_split_opt(x_train, y_train, min_samples_split_param, min_samples_leaf_param) # 得到min_samples_leaf和min_samples_split
        self.__max_features_opt(x_train, y_train, max_features_param) # 得到max_features
        self.model = self.model(n_estimators=self.n_estimators, criterion='gini', max_depth=self.max_depth, 
                    min_samples_split=self.min_samples_split, min_samples_leaf=self.min_samples_leaf, max_features=self.max_features,
                    max_leaf_nodes=self.max_leaf_nodes, min_impurity_decrease=self.min_impurity_decrease,  # bootstrap=True,
                    oob_score=self.oob_score__ , # n_jobs=None, 
                    random_state=self.random_state #, verbose=0,
                    # warm_start=False, class_weight=None, ccp_alpha=0, max_samples=None
                    )
        self.model.fit(x_train, y_train)
        self.classes = self.model.classes_
        self.base_estimators = self.model.estimator_
        self.feature_importances = self.model.feature_importances_
        self.n_features_in = self.model.n_features_in_
        self.oob_score = self.model.oob_score_
        print(f'oob_score:{self.oob_score}')

    def predict(self, x):
        y_pred = self.model.predict(x)
        print(y_pred)
        return y_pred
    
    def pred_acc(self, x, y):
        y_pred = self.predict(x)
        metric = self.metrics(y, y_pred)
        return f'{self.metrics.__name__}:{metric}'
    
    def plot_importance(self, x):
        sorted_index = self.feature_importances.argsort()
        plt.figure(figsize=(10,6))
        plt.barh(range(x.shape[1]), self.feature_importances[sorted_index])
        plt.yticks(np.arange(x.shape[1]), x.columns[sorted_index])
        plt.xlabel('Feature Importance')
        plt.ylabel('Feature')
        plt.title('Random Forest')
        plt.savefig('rf/rf_importances.jpg')
        # plt.show()




        


        