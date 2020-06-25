import scipy
import pandas as pd
import keras
from keras.models import Sequential
from keras.layers import Dense
from sklearn.model_selection import train_test_split
from sklearn import preprocessing
import numpy as np

def data():
    dir_name = 'E:/Desktop_Study/大三下学期/模式识别/大作业/data.csv'
    df = pd.read_csv(dir_name, encoding='gb18030')

# 数据清洗
    # 1)删除包含缺失值的行
    # df.dropna(inplace=True)
    # 1)缺失值填充
    df.fillna(0, inplace=True)
    # 2)将无穷大的数据转换为0
    train_inf = np.isinf(df)
    df[train_inf] = 0

    # 数据集构成：
    #   后n个为全部特征值
    #   前18个为label
    y = df.iloc[:, 0:18].values
    X = df.iloc[:, 18:48].values
    # 最小最大值标准化
    # min_max_scaler = preprocessing.MinMaxScaler()
    # X = min_max_scaler.fit_transform(X)

    # 绝对值最大标准化
    # max_abs_scaler = preprocessing.MaxAbsScaler()
    # X = max_abs_scaler.fit_transform(X)

    # 二值化–特征的二值化
    # binarizer = preprocessing.Binarizer()
    # X = binarizer.transform(X)

    # one-hot 编码
    enc = preprocessing.OneHotEncoder()
    enc.fit(X)
    X = enc.transform(X).toarray()
    print(X)
