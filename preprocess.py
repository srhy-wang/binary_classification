import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.utils.validation import column_or_1d

# 数据预处理——变量转换及标准化
def pretransX(x):
    try:
        x.drop(['oral'], axis=1, inplace = True)  #oral值没有变异性
        x['gender'] = x['gender'].apply(lambda x: 1 if x=='M' else 0)
        x['tartar'] = x['tartar'].apply(lambda x: 1 if x=='Y' else 0)

        a = x.shape[0]
        x = x[x['eyesight(left)']!=9.9] #删除异常值
        x = x[x['eyesight(right)']!=9.9]
        b = x.shape[0]

        print(f'delete {a-b} samples')
        return x
    
    except Exception as e:
        print(e)

def prescalarX(x): 
    # print(x.dtypes)
    try:
        scaler = StandardScaler() # 进行标准化
        x = scaler.fit_transform(x)
        return x
    
    except ValueError:
        print('Some variables have wrong dtypes!')

    except Exception as e:
        print(e)

def preprocessY(y, x_train):
    index = x_train.index
    y = y[y.index.isin(index)]
    y = column_or_1d(y)
    return y