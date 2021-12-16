# -*- coding: utf-8 -*-

"""
Random forest for time series classification

Author: Zhou Ya'nan
"""
import collections
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn import metrics
from sklearn.model_selection import cross_validate, train_test_split, GridSearchCV
import joblib
# import imblance_sample


def load_data(datafile):
    """Input:
    datafile: location of the data
    """
    all_data = np.loadtxt(datafile, delimiter=',')
    # np.random.shuffle(all_data)
    all_data = all_data[0:, 3:]  # 去掉第一列PlotID
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
    csv_data = csv_data.iloc[:, 3:]

    segment = csv_data.iloc[:, :-1].to_numpy()
    label = csv_data.iloc[:, -1].to_numpy()
    field = csv_data.columns.values

    print(np.shape(segment), np.shape(label))
    # print("Numbers of category：", collections.Counter(label))
    return segment, label, field


def train_model(train_x, train_y):
    print("### Model training searching")
    # 拆分为训练集和验证集
    train_x, valid_x, train_y, valid_y = train_test_split(train_x, train_y, test_size=0.2, random_state=100)

    # 类别不平衡处理
    # train_x, train_y = imblance_sample.smote_tomek_combine(train_x, train_y)

    # RandomForest分类器参数与分类器(不管任何参数，都用默认的)
    rfc = RandomForestClassifier(oob_score=True, random_state=10,
                                 n_estimators=100,
                                 max_depth=None,
                                 min_samples_split=2,
                                 min_samples_leaf=1,
                                 max_features="auto",
                                 )
    rfc.fit(train_x, train_y)
    print("### Training accuracy %g" % rfc.oob_score_)

    # 分类精度
    overall_accuracy = rfc.score(valid_x, valid_y)
    print("### Validation accuracy %g" % overall_accuracy)

    return rfc


def grid_search_parameter(train_x, train_y):
    print("### Model parameter sear")
    # 首先对n_estimators进行网格搜索,得到了最佳的弱学习器迭代次数
    param_test_estimators = {'n_estimators': list(range(10, 200, 10))}
    gsearch_estimators = GridSearchCV(estimator=RandomForestClassifier(min_samples_split=2,
                                                                       min_samples_leaf=1,
                                                                       max_depth=None,
                                                                       max_features="auto",
                                                                       # class_weight='balanced_subsample',
                                                                       random_state=10),
                                      param_grid=param_test_estimators, scoring='accuracy', cv=5)

    gsearch_estimators.fit(train_x, train_y)
    print(gsearch_estimators.cv_results_ )
    print(gsearch_estimators.best_params_)
    print(gsearch_estimators.best_score_)

    # 接着对决策树最大深度max_depth和内部节点再划分所需最小样本数min_samples_split进行网格搜索。
    param_test_depth_minsplit = {'max_depth': list(range(2, 20, 2)), 'min_samples_split': list(range(1, 21, 2))}
    gsearch_depth_minsplit = GridSearchCV(estimator=RandomForestClassifier(n_estimators=100,
                                                                           min_samples_leaf=1,
                                                                           max_features='auto',
                                                                           # class_weight='balanced_subsample',
                                                                           random_state=10),
                                          param_grid=param_test_depth_minsplit, scoring='accuracy', iid=False, cv=5)

    gsearch_depth_minsplit.fit(train_x, train_y)
    print(gsearch_depth_minsplit.cv_results_ )
    print(gsearch_depth_minsplit.best_params_)
    print(gsearch_depth_minsplit.best_score_)

    # 再对内部节点再划分所需最小样本数min_samples_split和叶子节点最少样本数min_samples_leaf一起调参
    param_test_minsplit_minleaf = {'min_samples_split': list(range(7, 21, 2)), 'min_samples_leaf': list(range(2, 20, 2))}
    gsearch_minsplit_minleaf = GridSearchCV(estimator=RandomForestClassifier(n_estimators=170, max_depth=15,
                                                                             max_features=25,
                                                                             class_weight='balanced_subsample',
                                                                             random_state=10),
                                            param_grid=param_test_minsplit_minleaf, scoring='accuracy', iid=False, cv=5)

    gsearch_minsplit_minleaf.fit(train_x, train_y)
    print(gsearch_minsplit_minleaf.cv_results_ )
    print(gsearch_minsplit_minleaf.best_params_)
    print(gsearch_minsplit_minleaf.best_score_)

    # 最后我们再对最大特征数max_features做调参:
    param_test_features = {'max_features': list(range(5, 40, 5))}
    gsearch_features = GridSearchCV(estimator=RandomForestClassifier(n_estimators=130,
                                                                     max_depth=15,
                                                                     min_samples_split=2,
                                                                     min_samples_leaf=2,
                                                                     class_weight='balanced_subsample',
                                                                     random_state=10),
                                    param_grid=param_test_features, scoring='accuracy', iid=False, cv=5)

    gsearch_features.fit(train_x, train_y)
    print(gsearch_features.cv_results_ )
    print(gsearch_features.best_params_)
    print(gsearch_features.best_score_)


def model_predict(model, test_x):
    print("### Model Predicting")
    # 测试分析
    test_y = model.predict(test_x)
    print("Numbers of category：", collections.Counter(test_y))

    print("RF predicting Over!!!")
    return test_y


def model_predict_proba(model, test_x, threshold=None):
    print("### Model Predicting Proba")
    # 测试分析
    test_proba = model.predict_proba(test_x)

    if threshold:
        assert (test_proba.shape[1] == len(threshold))
        test_y = np.copy(test_proba)
        test_y[test_y < threshold] = 0
        test_y = np.argmax(test_y, axis=1)
        test_y = model.classes_[test_y]

    print("RF predicting Over!!!")
    return test_proba, test_y


def merge_prediction(templete_file, result_file, merge_file):

    templete_pd = pd.read_csv(templete_file, header=0, encoding="gbk")
    result_pd = pd.read_csv(result_file, header=0, encoding="gbk")

    merge_pd = pd.concat([templete_pd, result_pd], axis=1, ignore_index=False)
    merge_pd.to_csv(merge_file, sep=',')


def plot_importance(model, fields):
    """

    :param model:
    :param fields:
    :return:
    """
    importances = model.feature_importances_
    indices = np.argsort(importances)

    plt.title('Feature Importances')
    plt.barh(range(len(indices)), importances[indices], color='b', align='center')
    plt.yticks(range(len(indices)), [fields[i] for i in indices])
    plt.xlabel('Relative Importance')
    plt.show()
    pass


def main1():
    print("### RF classifier ###########################################")
    # 读取训练数据
    train_data = 'J:/FF/application_dataset/chongqing_agir/sample/attri1.csv'
    train_x, train_y, field = load_csvdata_pandas(train_data)

    # 数据标准化 & 训练模型 (本案例中不需要标准化，外部已经标准化过)
    # data_scaler = StandardScaler()
    # train_x = data_scaler.fit_transform(train_x)

    rfc_model = train_model(train_x, train_y)

    # 模型保存与载入
    # joblib.dump(rfc_model, 'saveModel/model.m')
    # rfc_model = joblib.load('saveModel/model.m')

    # 模型预测
    test_data = 'J:/FF/application_dataset/chongqing_agir/att000.csv'
    test_result = 'J:/FF/application_dataset/chongqing_agir/att000_result.csv'
    test_x, _, _ = load_csvdata_pandas(test_data)

    # test_x = data_scaler.transform(test_x)
    test_y = model_predict(rfc_model, test_x)
    np.savetxt(test_result, test_y.astype(int), delimiter=',')

    # 合并结果
    result_file = 'J:/FF/application_dataset/chongqing_agir/att000_result.csv'
    templete_file = 'J:/FF/application_dataset/chongqing_agir/att_result_templete.csv'
    merge_file = 'J:/FF/application_dataset/chongqing_agir/att000_result_merge.csv'
    merge_prediction(templete_file, result_file, merge_file)

    print("### Task over ###########################################")


def main2():
    print("### RF classifier ###########################################")
    # 读取训练数据
    train_data = 'I:/FF/application_dataset/chongqing_agir/MODIS2021/att_sample_result/sample/sample_att.csv'
    train_x, train_y, field = load_csvdata_pandas(train_data)

    # 数据标准化 & 训练模型 (本案例中不需要标准化，外部已经标准化过)
    # data_scaler = StandardScaler()
    # train_x = data_scaler.fit_transform(train_x)

    # grid_search_parameter(train_x, train_y)
    rfc_model = train_model(train_x, train_y)

    # 模型保存与载入
    # joblib.dump(rfc_model, 'saveModel/model.m')
    # rfc_model = joblib.load('saveModel/model.m')

    # 模型预测
    test_data = 'I:/FF/application_dataset/chongqing_agir/MODIS2021/att_sample_result/att.csv'
    test_result = 'I:/FF/application_dataset/chongqing_agir/MODIS2021/att_sample_result/att_result.csv'
    test_x, _, _ = load_csvdata_pandas(test_data)

    # test_x = data_scaler.transform(test_x)
    test_y = model_predict(rfc_model, test_x)
    np.savetxt(test_result, test_y.astype(int), delimiter=',')

    # 合并结果
    result_file = test_result
    templete_file = 'I:/FF/application_dataset/chongqing_agir/MODIS2021/att_sample_result/att_result_templete.csv'
    merge_file = 'I:/FF/application_dataset/chongqing_agir/MODIS2021/att_sample_result/att_result_merge.csv'
    merge_prediction(templete_file, result_file, merge_file)

    print("### Task over ###########################################")


if __name__ == "__main__":
    main2()
