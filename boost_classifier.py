# -*- coding: utf-8 -*-

"""
Boost for time series classification

Author: Zhou Ya'nan
"""
import numpy as np
import collections
import xgboost as xgb
from sklearn import metrics
from sklearn.model_selection import train_test_split
import imblance_sample


####################################################################
def load_data(datafile):
    all_data = np.loadtxt(datafile, delimiter=',')
    # np.random.shuffle(all_data)
    all_data = all_data[0:, 1:]  # 去掉第一列PlotID
    segment = all_data[0:, 1:]
    label = all_data[0:, 0]

    return segment, label


####################################################################
# 导入数据
train_data = 'G:/experiments-dataset/guiyang-crop/007-classification/plots-sample.csv'
train_x, train_y = load_data(train_data)
print("TRAIN shape: ", np.shape(train_x), np.shape(train_y))

train_x -= np.mean(train_x, axis=0)  # zero-center
train_x /= np.std(train_x, axis=0)  # normalize


'''
####################################################################
train_x, test_x, train_y, test_y = train_test_split(train_x, train_y, test_size=0.2, random_state=0)
# scikit-learn XGBoost分类器
model = xgb.XGBClassifier(max_depth=5, learning_rate=0.1, n_estimators=160, silent=True, objective='multi:softmax')
model.fit(train_x, train_y)
pred_y = model.predict(test_x)
print('ACC: %.4f' % metrics.accuracy_score(test_y, pred_y))
'''


####################################################################
# 拆分为训练集和验证集
train_x, valid_x, train_y, valid_y = train_test_split(train_x, train_y, test_size=0.2, random_state=100)

# 类别不平衡处理
#train_x, train_y = imblance_sample.smote_tomek_combine(train_x, train_y)

# XGBoost分类器
xg_train = xgb.DMatrix(train_x, label=train_y)
xg_valid = xgb.DMatrix(valid_x, label=valid_y)

# setup parameters for xgboost
params = {
    'booster': 'gbtree',
    'objective': 'multi:softmax',   # 多分类的问题
    'num_class': 6,                 # 类别数，与 multisoftmax 并用
    'gamma': 0.2,                   # 用于控制是否后剪枝的参数,越大越保守，一般0.1、0.2这样子。
    'max_depth': 12,                # 构建树的深度，越大越容易过拟合
    'lambda': 2,                    # 控制模型复杂度的权重值的L2正则化项参数，参数越大，模型越不容易过拟合。
    'subsample': 0.7,               # 随机采样训练样本
    'colsample_bytree': 0.7,        # 生成树时进行的列采样
    'min_child_weight': 3,
    'max_delta_step': 5,            # 如果在逻辑回归中类极其不平衡这时候他有可能会起到帮助作用
    'silent': 1,                    # 设置成1则没有运行信息输出，最好是设置为0.
    'eta': 0.007,                   # 如同学习率
    'seed': 1000,
    'nthread': 8,                   # cpu 线程数
}
watchlist = [(xg_train, 'train'), (xg_valid, 'valid')]

bst = xgb.train(params, xg_train, num_boost_round=1000, evals=watchlist)

valid_y_pred = bst.predict(xg_valid)

print('ACC: %.4f' % metrics.accuracy_score(valid_y, valid_y_pred))
# print('AUC: %.4f' % metrics.roc_auc_score(test_y, y_pred))
# print('Recall: %.4f' % metrics.recall_score(test_y, y_pred))
# print('F1-score: %.4f' % metrics.f1_score(test_y, y_pred))
# print('Precesion: %.4f' % metrics.precision_score(test_y, y_pred))
# print(metrics.confusion_matrix(test_y, y_pred))


####################################################################
# 预测分析
test_data = 'G:/experiments-dataset/guiyang-crop/007-classification/plots-test.csv'
predict_data = 'G:/experiments-dataset/guiyang-crop/007-classification/plots-test-result.csv'

testdata_x, testdata_y = load_data(test_data)
print(np.shape(testdata_x), np.shape(testdata_y))

testdata_x -= np.mean(testdata_x, axis=0)
testdata_x /= np.std(testdata_x, axis=0)

xg_test = xgb.DMatrix(testdata_x, label=testdata_y)
predict_y = bst.predict(xg_test)
np.savetxt(predict_data, predict_y, delimiter=',')
print("Numbers of category：", collections.Counter(predict_y))

print("Boost Crop Over!!!")
