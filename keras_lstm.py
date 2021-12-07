# -*- coding: utf-8 -*-

"""
Keras_LSTM for time series classification

Author: Zhou Ya'nan
"""
from __future__ import print_function
import argparse
import numpy as np
from keras.utils import np_utils
from keras.models import Sequential
from keras.layers import LSTM, Activation, Dense, Dropout, BatchNormalization
from keras.optimizers import Adam, SGD
from keras.callbacks import ReduceLROnPlateau
from keras.utils.vis_utils import plot_model
import data_utils
import imblance_sample


####################################################################
# 绘制学习曲线
def learning_curves(history):
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(2, 1, figsize=(10, 10))
    ax[0].plot(history.history['loss'], color='r', label='Training Loss')
    ax[0].plot(history.history['val_loss'], color='g', label='Validation Loss')
    ax[0].legend(loc='best', shadow=True)
    ax[0].grid(True)

    ax[1].plot(history.history['acc'], color='r', label='Training Accuracy')
    ax[1].plot(history.history['val_acc'], color='g', label='Validation Accuracy')
    ax[1].legend(loc='best', shadow=True)
    ax[1].grid(True)
    plt.show()


# 学习速度递减函数
learning_rate_reduction = ReduceLROnPlateau(monitor='val_acc', patience=3, verbose=1, factor=0.8, min_lr=0.00001)


####################################################################
# 设置模型的超参数
batch_size = 64     # 每块训练样本数
n_feature = 2       # 每个时刻的输入特征是2维的，多变量的个数
n_step = 28         # 步长,时序长度
n_hidden = 32       # 隐含层的数量
n_classes = 6       # 类别
learn_rate = 0.0001 # 学习率
drop_rat = 0.4      # 丢失率
epochs = 49         # 迭代次数

# x标准化到0-1; y使用one-hot  输入 nxm的矩阵 每行m维切成n个输入
# train_y = np_utils.to_categorical(train_y, num_classes=n_classes)
# test_y = np_utils.to_categorical(test_y, num_classes=n_classes)


def read_traindata(training_file):

    ####################################################################
    # 导入数据
    train_x, train_y = data_utils.load_data(training_file)
    print("Training data {}, label {}".format(np.shape(train_x), np.shape(train_y)))

    # 类别不平衡处理
    # train_x = np.reshape(train_x, [np.shape(train_x)[0], -1])
    # train_x, train_y = imblance_sample.smote_tomek_combine(train_x, train_y)
    # train_x = np.reshape(train_x, [np.shape(train_x)[0], -1, 2])

    # zero-center,但越来越不重要
    # train_x -= np.mean(train_x, axis=0)  # zero-center
    # train_x /= np.std(train_x, axis=0)  # normalize
    # one_hot转换
    train_y = np_utils.to_categorical(train_y, num_classes=n_classes)

    print("### Reading training data over")
    return train_x, train_y


def train_lstm(train_x, train_y):

    ####################################################################
    #  创建LSTM模型
    model = Sequential(name='Keras-LSTM-Crop')
    model.add(LSTM(n_hidden, return_sequences=True, batch_input_shape=(None, n_step, n_feature), unroll=True))
    model.add(LSTM(n_hidden, return_sequences=True))
    model.add(LSTM(n_hidden, return_sequences=True))
    model.add(LSTM(n_hidden))
    model.add(Dropout(rate=drop_rat))
    model.add(BatchNormalization())
    model.add(Dense(n_classes, activation='softmax'))
    # model.add(Activation('softmax'))

    model.summary()
    # 显示网络结构图
    # plot_model(model, to_file='./model.png', show_shapes=True, show_layer_names=True)

    model.compile(optimizer=Adam(lr=learn_rate), loss='categorical_crossentropy', metrics=['accuracy'])
    training_history = model.fit(train_x, train_y, batch_size=batch_size, epochs=epochs, validation_split=0.2,
                                 shuffle=True, verbose=1, class_weight='balanced',
                                 callbacks=[learning_rate_reduction])

    # 显示训练精度和衰减
    # learning_curves(training_history)

    print("### Training model over")
    return model


def read_testdata(test_file):
    ####################################################################
    # 导入数据
    test_x, test_y = data_utils.load_data(test_file)
    print("Test data {}".format(np.shape(test_x)))

    # test_x -= np.mean(test_x, axis=0)
    # test_x /= np.std(test_x, axis=0)

    print("### Reading test data over")
    return test_x


def lstm_predict(model, test_x):

    ####################################################################
    # 预测
    test_results = model.predict(test_x)
    # test_results = data_utils.argmax_label(test_results) + 1

    print("### Model predicting data over")
    return test_results


def main():
    print("### lstm.main() ###########################################")

    parser = argparse.ArgumentParser(description='Manual to this LSTM-based classification')
    parser.add_argument('--training_file', type=str, default=None, help='data file in csv for training')
    parser.add_argument('--test_file', type=str, default=None, help='data file in csv for testing')
    parser.add_argument('--predict_file', type=str, default=None, help='data file in csv for testing results')
    parser.add_argument('--time_step', type=int, default=32)
    parser.add_argument('--class_number', type=int, default=6)
    parser.add_argument('--epochs', type=int, default=99)
    args = parser.parse_args()

    training_file = args.training_file
    test_file = args.test_file
    predict_file = args.predict_file
    if training_file is None or test_file is None or predict_file is None:
        print("### Wrong parameters ###")
        return False

    global n_step; global n_classes; global epochs
    n_step = args.time_step
    n_classes = args.class_number
    epochs = args.epochs

    # training_file = 'G:/experiments-dataset/guiyang-crop/007-classification/plots-sample.csv'
    # test_file = 'G:/experiments-dataset/guiyang-crop/007-classification/plots-test.csv'
    # predict_file = 'G:/experiments-dataset/guiyang-crop/007-classification/plots-test-result.csv'

    train_x, train_y = read_traindata(training_file)
    lstm_model = train_lstm(train_x, train_y)

    test_x = read_testdata(test_file)
    test_results = lstm_predict(lstm_model, test_x)
    test_results = data_utils.argmax_label(test_results)
    np.savetxt(predict_file, test_results, delimiter=',')

    print("Keras-LSTM-Crop Over!!!")
    return True


####################################################################
if __name__ == "__main__":
    main()
