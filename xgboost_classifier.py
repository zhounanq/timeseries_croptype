# -*- coding: utf-8 -*-

"""
Random forest for time series classification

Author: Zhou Ya'nan
"""
import collections
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from xgboost import XGBClassifier
from xgboost import plot_importance
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn import metrics
from sklearn.model_selection import cross_validate, train_test_split, GridSearchCV
# import imblance_sample


def load_data(datafile):
    """Input:
    datafile: location of the data
    """
    all_data = np.loadtxt(datafile, delimiter=',')
    # np.random.shuffle(all_data)
    all_data = all_data[0:, 1:]  # 去掉第一列PlotID
    segment = all_data[0:, 1:]
    label = all_data[0:, 0]

    print(np.shape(segment), np.shape(label))
    print("Numbers of category：", collections.Counter(label))
    return segment, label


def load_csvdata_pandas(datafile):
    """

    :param datafile:
    :return:
    """
    csv_data = pd.read_csv(datafile, header=0, encoding="gbk")
    csv_data = csv_data.iloc[:, 2:]

    segment = csv_data.iloc[:, :-1].to_numpy()
    label = csv_data.iloc[:, -1].to_numpy()

    print(np.shape(segment), np.shape(label))
    print("Numbers of category：", collections.Counter(label))
    return segment, label


def train_model(train_x, train_y):

    # 拆分为训练集和验证集
    train_x, valid_x, train_y, valid_y = train_test_split(train_x, train_y, test_size=0.2, random_state=100)

    # 类别不平衡处理
    # train_x, train_y = imblance_sample.smote_tomek_combine(train_x, train_y)

    # XGBoost分类器
    xgb = XGBClassifier(
        max_depth=6,            # 构建树的深度，越大越容易过拟合
        learning_rate=0.3,      # 学习率
        n_estimators=100,       # 树的个数

        # objective = 'multi:softmax', # 多分类问题，指定学习任务和响应的学习目标
        # reg:linear–线性回归；
        # reg:logistic–逻辑回归；
        # binary:logistic –二分类的逻辑回归问题，输出为概率；
        # multi:softmax –让XGBoost采用softmax目标函数处理多分类问题，同时需要设置参数num_class（类别个数）
        # num_class = 10,       # 类别数，多分类与multisoftmax并用
        booster='gbtree',       # 指定弱学习器的类型，默认值为gbtree使用基于树的模型进行计算。可选gblinear线性模型作为弱学习器。

        gamma=0,                # 树的叶子节点上做进一步分区所需的最小损失减少，越大越保守，一般0.1 0.2这样子
        min_child_weight=1,
        # 这个参数默认为1，是每个叶子里面h的和至少是多少，对正负样本不均衡时的0-1分类而言
        # 假设h在0.01附近，min_child_weight为1 意味着叶子节点中最少需要包含100个样本
        # 这个参数非常影响结果，控制叶子节点中二阶导的和的最小值，该参数值越小，越容易过拟合
        max_delta_step=0,       # 最大增量步长，我们允许每个树的权重估计
        subsample=1,            # 随机采样训练样本，训练实例的子采样比

        colsample_bytree=1,     # 生成树时进行的列采样
        # reg_alpha=0,          # L1正则项参数
        reg_lambda=1,           # 控制模型复杂度的权重值的L2正则化项参数，参数越大，模型越不容易过拟合

        # scale_pos_weight =1 # 如果取值大于0的话，在类别样本不平衡的情况下有助于快速收敛，平衡正负权重

        # eval_metric='mlogloss', # 评估指标，可以传递各种评估方法组成的list；rmse回归任务、mlogloss多分类任务、error二分类任务、auc二分类任务

        silent=0,               # 设置成1则没有运行信息输出，最好是设置为0，是否在运行升级时打印消息
        # nthread=4             # CPU 线程数默认最大
        seed = 1000             # 随机种子
    )
    xgb.fit(train_x, train_y)
    # print("### Training accuracy %g" % xgb.oob_score_)

    # 分类精度
    overall_accuracy = xgb.score(valid_x, valid_y)
    print("### Validation accuracy %g" % overall_accuracy)

    # 显示重要特征
    plot_importance(xgb)
    plt.show()

    return xgb


def model_predict(model, test_x):

    # 测试分析
    test_y = model.predict(test_x)
    print("Numbers of category：", collections.Counter(test_y))

    print("xgboost predicting Over!!!")
    return test_y


def main():
    print("### xgboost classifier main() ###########################################")
    # 读取训练数据
    train_data = 'J:/FF/application_dataset/chongqing_agir/sample/attri12.csv'
    train_x, train_y = load_csvdata_pandas(train_data)

    # 数据标准化 & 训练模型 (本案例中不需要标准化，外部已经标准化过)
    # data_scaler = StandardScaler()
    # train_x = data_scaler.fit_transform(train_x)

    xgb_model = train_model(train_x, train_y)

    # 模型预测
    test_data = 'G:/common-dataset/Sentinel-1/weining-crop/plots/2/03/wndikuai2-test.csv'
    test_result = 'G:/common-dataset/Sentinel-1/weining-crop/plots/2/03/wndikuai2-result.csv'
    test_x, _ = load_csvdata_pandas(test_data)

    # test_x = data_scaler.transform(test_x)
    test_y = model_predict(xgb_model, test_x)

    np.savetxt(test_result, test_y, delimiter=',')
    print("### Task over ###########################################")


if __name__ == "__main__":
    main()
