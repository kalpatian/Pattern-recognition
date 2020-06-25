import scipy
import pandas as pd
import keras
from keras.models import Sequential
from keras.layers import Dense
from sklearn.model_selection import train_test_split
from sklearn import preprocessing
import numpy as np
import argparse

parser = argparse.ArgumentParser("Generative Adversarial Examples")
parser.add_argument('--loss_func', type=str, default='binary', help="binary | categorical")
parser.add_argument('--datapath', type=str, default=None, help="input data path")
parser.add_argument('--savepath', type=str, default=None, help="save path for the model checkpoint")
parser.add_argument('--normalization', type=str, default='onehot', help='onehot | minmax | maxabs | binarizer')
args = parser.parse_args()


def deep_model(feature_dim, label_dim, loss_func):
    model = Sequential()
    print("create model. feature_dim ={}, label_dim ={}".format(feature_dim, label_dim))
    model.add(Dense(100, activation='sigmoid', input_dim=feature_dim))
    model.add(Dense(100, activation='relu'))
    model.add(Dense(label_dim, activation='sigmoid'))
    if loss_func == 'binary':
        model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    else:
        model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

    return model


def train_deep(X_train, y_train, X_test, y_test, savepath, loss_func):
    feature_dim = X_train.shape[1]
    label_dim = y_train.shape[1]
    model = deep_model(feature_dim, label_dim, loss_func)
    model.summary()
    model.fit(X_train, y_train, batch_size=64, epochs=5, validation_data=(X_test, y_test))
    loss, accuracy = model.evaluate(X_test, y_test)
    print('\ntest loss', loss)
    print('accuracy', accuracy)
    model.save(savepath)


if __name__ == '__main__':
    norm = args.normalization

    df = pd.read_csv(args.datapath, encoding='gb18030')
    # 数据清洗
    # 1)删除包含缺失值的行
    # df.dropna(inplace=True)
    # 1)缺失值填充
    df.fillna(0, inplace=True)
    # 2)将无穷大的数据转换为0
    # train_inf = np.isinf(df)
    # df[train_inf] = 0

    # 数据集构成：
    #   后n个为全部特征值
    #   前18个为label
    y = df.iloc[:, 6:24].values
    X = df.iloc[:, 0:6].values
    if norm == 'minmax':
        # 最小最大值标准化
        min_max_scaler = preprocessing.MinMaxScaler()
        X = min_max_scaler.fit_transform(X)

    elif norm == 'maxabs':
        # 绝对值最大标准化
        max_abs_scaler = preprocessing.MaxAbsScaler()
        X = max_abs_scaler.fit_transform(X)

    elif norm == 'binarizer':
        # 二值化–特征的二值化
        binarizer = preprocessing.Binarizer()
        X = binarizer.transform(X)

    elif norm == 'onehot':
        # one-hot 编码
        enc = preprocessing.OneHotEncoder()
        enc.fit(X)
        X = enc.transform(X).toarray()

    else:
        enc = preprocessing.OneHotEncoder()
        enc.fit(X)
        X = enc.transform(X).toarray()

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)
    y_train = y_train.astype(np.float64)
    y_test = y_test.astype(np.float64)
    savepath = args.savepath + norm + '_' + args.loss_func + '_model2.h5'
    train_deep(X_train, y_train, X_test, y_test, savepath, args.loss_func)
