# -*- coding: utf-8 -*-

"""
sampling for imblance samples, see https://blog.csdn.net/qq_31813549/article/details/79964973

Author: Zhou Ya'nan
"""
import numpy as np
from collections import Counter
from imblearn.over_sampling import RandomOverSampler
from imblearn.over_sampling import SMOTE, ADASYN
from imblearn.under_sampling import ClusterCentroids
from imblearn.under_sampling import RandomUnderSampler
from imblearn.under_sampling import NearMiss

from imblearn.combine import SMOTEENN
from imblearn.combine import SMOTETomek
from imblearn.ensemble import EasyEnsemble
from imblearn.ensemble import BalanceCascade
from imblearn.ensemble import BalancedBaggingClassifier

from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
from sklearn.ensemble import BaggingClassifier
from sklearn.tree import DecisionTreeClassifier


####################################################################
## oversample
####################################################################
def random_oversample(forx, fory):
    """ Function to oversample randomly """
    ros = RandomOverSampler(random_state=0)
    res_x, res_y = ros.fit_sample(forx, fory)

    print("Before: ", sorted(Counter(fory).items()))
    print("After: ", sorted(Counter(res_y).items()))
    return res_x, res_y


def smote_oversample(forx, fory):
    """ Function to Synthetic Minority Oversampling Technique """
    kind = 'regular'  # regular, borderline1; borderline2; svm
    smote = SMOTE(kind=kind)
    res_x, res_y = smote.fit_sample(forx, fory)

    print("Before: ", sorted(Counter(fory).items()))
    print("After: ", sorted(Counter(res_y).items()))
    return res_x, res_y


def adasyn_oversample(forx, fory):
    """ Function to Adaptive Synthetic """
    adasyn = ADASYN()
    res_x, res_y = adasyn.fit_sample(forx, fory)

    print("Before: ", sorted(Counter(fory).items()))
    print("After: ", sorted(Counter(res_y).items()))
    return res_x, res_y


####################################################################
## undersample
####################################################################
def random_undersample(forx, fory):
    """ Function to undersample randomly """
    rus = RandomUnderSampler(random_state=0, replacement=False)
    res_x, res_y = rus.fit_sample(forx, fory)

    print("Before: ", sorted(Counter(fory).items()))
    print("After: ", sorted(Counter(res_y).items()))
    return res_x, res_y


def clustercentroids_undersample(forx, fory):
    """ Function to Cluster Centroids """
    cc = ClusterCentroids(random_state=0)
    res_x, res_y = cc.fit_sample(forx, fory)

    print("Before: ", sorted(Counter(fory).items()))
    print("After: ", sorted(Counter(res_y).items()))
    return res_x, res_y


def nearmiss_undersample(forx, fory):
    """ Function to NearMiss """
    ver = 1  # 1,2,3
    nm = NearMiss(random_state=0, version=ver)
    res_x, res_y = nm.fit_sample(forx, fory)

    print("Before: ", sorted(Counter(fory).items()))
    print("After: ", sorted(Counter(res_y).items()))
    return res_x, res_y


####################################################################
### combine
####################################################################
def smote_enn_combine(forx, fory):
    """ Function to  """
    smote_enn = SMOTEENN(random_state=0)
    res_x, res_y = smote_enn.fit_sample(forx, fory)

    print("Before: ", sorted(Counter(fory).items()))
    print("After: ", sorted(Counter(res_y).items()))
    return res_x, res_y


def smote_tomek_combine(forx, fory):
    """ Function to  """
    smote_tomek = SMOTETomek(random_state=0)
    res_x, res_y = smote_tomek.fit_sample(forx, fory)

    print("Before: ", sorted(Counter(fory).items()))
    print("After: ", sorted(Counter(res_y).items()))
    return res_x, res_y


####################################################################
### ensemble (下面两个还不知道如何使用呢！)
####################################################################
def easy_ensemble(forx, fory):
    """ Function to  """
    ee = EasyEnsemble(random_state=0, n_subsets=10)
    res_x, res_y = ee.fit_sample(forx, fory)

    print(res_x.shape)
    print("After: ", sorted(Counter(res_y[0]).items()))
    return res_x, res_y


def balancecascade_ensemble(forx, fory):
    """ Function to  """
    bc = BalanceCascade(random_state=0, estimator=LogisticRegression(random_state=0), n_max_subset=4)
    res_x, res_y = bc.fit_sample(forx, fory)

    print(res_x.shape)
    print("After: ", sorted(Counter(res_y).items()))
    return res_x, res_y

    ### BalancedBaggingClassifier 允许在训练每个基学习器之前对每个子集进行重抽样
