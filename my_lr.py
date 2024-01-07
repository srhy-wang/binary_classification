from sklearn.linear_model import LogisticRegressionCV
import sklearn.metrics as skm
from sklearn.model_selection import cross_val_score
import matplotlib.pyplot as plt
import numpy as np


class Mylogisticregression:
    def __init__(self, Cs=10, fit_intercept=True, cv=10, 
                penalty='l2', scoring=None, solver='saga', tol=0.0001, 
                max_iter=200, class_weight=None, n_jobs=None,
                multi_class='ovr', 
                metrics='accuracy'):
        self.Cs=Cs
        self.fit_intercept=fit_intercept
        self.cv=cv
        self.penalty=penalty
        self.scoring=scoring
        self.solver=solver
        self.tol=tol
        self.max_iter=max_iter
        self.class_weight=class_weight
        self.n_jobs=n_jobs,
        self.multi_class=multi_class
        self.model = LogisticRegressionCV
        self.classes = None
        self.coef = None
        self.intercept = None
        
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
        
    def __cs_opt(self, x, y, cs_params): ## 内部方法 外部不可调用
        best_param = 0
        scores = []
        lambdas = cs_params
        for i in lambdas:
            lr = self.model(Cs=i)
            score = cross_val_score(lr, x, y, cv=10, scoring='accuracy')
            scores.append(np.mean(score))
            if np.mean(score)>=max(scores):
                best_param = i
        plt.figure(figsize=(8,8))
        plt.plot(lambdas, scores)
        plt.xlabel('cs', fontsize=12)
        plt.ylabel('accuracy', fontsize=12)
        plt.savefig('lr/lr_param.jpg')
        # plt.show()
        self.cs = best_param
        print(f'cs:{self.cs}')

    def fit(self, x_train, y_train, cs_params):
        self.__cs_opt(x_train, y_train, cs_params)
        self.model  = self.model(Cs=self.cs,fit_intercept=self.fit_intercept, cv=self.cv, 
                                    penalty=self.penalty, scoring=self.scoring, solver=self.solver, tol=self.tol, 
                                    max_iter=self.max_iter, class_weight=self.class_weight, #n_jobs=self.n_jobs,
                                    multi_class=self.multi_class)
        self.model = self.model.fit(x_train, y_train)
        self.classes = self.model.classes_
        self.coef = self.model.coef_
        self.intercept = self.model.intercept_
    
    def predict(self, x):
        y_pred = self.model.predict(x)
        return y_pred

    def pred_acc(self, x, y):
        # y = column_or_1d(y)
        y_pred = self.predict(x)
        metric = self.metrics(y ,y_pred)
        return f'{self.metrics.__name__}:{metric}'