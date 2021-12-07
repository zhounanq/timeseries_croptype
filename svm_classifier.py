# -*- coding: utf-8 -*-

"""
SVM for time series classification

Author: Zhou Ya'nan
"""
import collections
import numpy as np
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn import metrics
from sklearn.model_selection import train_test_split, GridSearchCV, cross_validate
import imblance_sample


def load_data(datafile):
    all_data = np.loadtxt(datafile, delimiter=',')
    # np.random.shuffle(all_data)
    all_data = all_data[0:, 1:]  # 去掉第一列PlotID
    segment = all_data[0:, 1:]
    label = all_data[0:, 0]

    return segment, label


####################################################################
# 导入数据
train_data = 'G:/common-dataset/Sentinel-1/weining-crop/plots/1/03/weiningdikuai-sample.csv'
train_x, train_y = load_data(train_data)
print("TRAIN shape: ", np.shape(train_x), np.shape(train_y))
print("Numbers of category：", collections.Counter(train_y))

# 数据标准化
scaler = StandardScaler()
train_x = scaler.fit_transform(train_x)


####################################################################
# 拆分为训练集和验证集
train_x, test_x, train_y, test_y = train_test_split(train_x, train_y, test_size=0.2, random_state=100)

# 类别不平衡处理
# train_x, train_y = imblance_sample.smote_tomek_combine(train_x, train_y)

# 分类器参数与分类器（其中参数class_weight和C用于控制不平衡类别分类）
clf = SVC(cache_size=512, class_weight='balanced', C=10, gamma='auto', probability=True, decision_function_shape="ovr")
clf.fit(train_x, train_y)

# 分类精度
overall_accuracy = clf.score(test_x, test_y)
print("### Validation accuracy %g" % overall_accuracy)

"""
# Grid Search for parameters
param_search = {'C': [1, 10, 20, 50, 100], 'gamma': list(np.arange(0.01, 0.4, 0.05))}
gsearch_c_gamma = GridSearchCV(estimator=SVC(cache_size=512, class_weight='balanced',
                                             probability=True, decision_function_shape="ovr"),
                               param_grid=param_search, scoring='accuracy', iid=False, cv=5)
gsearch_c_gamma.fit(train_x, train_y)
print(gsearch_c_gamma.grid_scores_)
print(gsearch_c_gamma.best_params_)
print(gsearch_c_gamma.best_score_)
"""


####################################################################
# 测试分析
test_data = 'G:/experiments-dataset/guiyang-crop/007-classification/plots-test.csv'
predict_data = 'G:/experiments-dataset/guiyang-crop/007-classification/plots-test-result.csv'

# 读入数据+标准化
testdata_x, testdata_y = load_data(test_data)
print(np.shape(testdata_x), np.shape(testdata_y))
testdata_x = scaler.transform(testdata_x)

# 预测分类+写出
testdata_y = clf.predict(testdata_x)
np.savetxt(predict_data, testdata_y, delimiter=',')
print("Numbers of category：", collections.Counter(testdata_y))

print("SVM Crop Over!!!")
