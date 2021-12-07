# -*- coding: utf-8 -*-

"""
Balanced Bagging Classifier for time series classification

Author: Zhou Ya'nan
"""
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn import model_selection, metrics
from sklearn.model_selection import train_test_split
from imblearn.ensemble import BalancedBaggingClassifier


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
# train_data = 'G:/experiments-dataset/guiyang-crop/004-classification/plots-sample-labels-re.csv'
train_data = 'G:/experiments-dataset/guiyang-crop/004-classification/plots-sample.csv'
train_x, train_y = load_data(train_data)
print("TRAIN shape: ", np.shape(train_x), np.shape(train_y))

train_x -= np.mean(train_x, axis=0)  # zero-center
train_x /= np.std(train_x, axis=0)  # normalize


####################################################################
# 拆分为训练集和验证集
train_x, valid_x, train_y, valid_y = train_test_split(train_x, train_y, test_size=0.4, random_state=100)
# 分类器参数与分类器
bbc = BalancedBaggingClassifier(base_estimator=RandomForestClassifier(),
                                ratio='auto',
                                replacement=False,
                                random_state=0)
bbc.fit(train_x, train_y)

# 分类精度
overall_accuracy = bbc.score(valid_x, valid_y)
print("### Validation accuracy %g" % overall_accuracy)


####################################################################
# 测试分析
test_data = 'G:/experiments-dataset/guiyang-crop/004-classification/plots-test.csv'
predict_data = 'G:/experiments-dataset/guiyang-crop/004-classification/plots-test-result.csv'

testdata_x, testdata_y = load_data(test_data)
print(np.shape(testdata_x), np.shape(testdata_y))
testdata_x -= np.mean(testdata_x, axis=0)  # zero-center
testdata_x /= np.std(testdata_x, axis=0)  # normalize

testdata_y = bbc.predict(testdata_x)
np.savetxt(predict_data, testdata_y, delimiter=',')


print("SVM Crop Over!!!")



