# -*- coding: utf-8 -*-

"""
LSTM for time series classification

Author: Zhou Ya'nan
"""
import numpy as np
import tensorflow as tf
from tensorflow.contrib import rnn
import data_utils

# 设置 GPU 按需增长
config = tf.ConfigProto()
config.gpu_options.allow_growth = True
sess = tf.Session(config=config)

tf.reset_default_graph()

# 设置变量
train_data = 'F:/experiment-dataset/03-lstm-sar-crop/data/train_ts22.csv'
test_data = 'F:/experiment-dataset/03-lstm-sar-crop/data/test_ts22.csv'
tbdir = "F:/experiment-dataset/03-lstm-sar-crop/data/log"


####################################################################
# 导入数据
multival_size = 2
train_x, train_y = data_utils.load_data(train_data)
print(np.shape(train_x), np.shape(train_y))

# zero-center,但越来越不重要
train_x -= np.mean(train_x, axis=0)  # zero-center
train_x /= np.std(train_x, axis=0)  # normalize


test_x, test_y = data_utils.load_data(test_data)
print(np.shape(test_x))
test_x -= np.mean(test_x, axis=0)
test_x /= np.std(test_x, axis=0)


####################################################################
# 设置模型的超参数
batch_size_num = 128
max_iterations = 10000
lr = 0.0001         # 训练速度
input_size = 2      # 每个时刻的输入特征是2维的，多变量的个数
time_size = 31      # 时序长度为31
hidden_size = 128   # 隐含层的数量
layer_num = 4       # LSTM layer 的层数
class_num = 7       # 最后输出分类类别数量(回归预测为1)


####################################################################
# 定义模型的输入与标记
with tf.name_scope('inputs'):
    model_inputs = tf.placeholder(tf.float32, [None, time_size, input_size], name='x_input')
    model_labels = tf.placeholder(tf.float32, [None, class_num], name='y_input')
    # 在训练和测试的时候，用不同的 batch_size.所以采用占位符的方式
    batch_size = tf.placeholder(tf.int32, [], name='batch_size_input')
    keep_prob = tf.placeholder(tf.float32, [], name='keep_prob_input')


# 实现 RNN/LSTM 网络
####################################################################
# **步骤2：创建多层lstm(在1.2.1版本中创建)
def lstm_cell():
    cell = rnn.LSTMCell(hidden_size, reuse=tf.get_variable_scope().reuse)
    with tf.name_scope('lstm_dropout'):
        return rnn.DropoutWrapper(cell, output_keep_prob=keep_prob)
with tf.name_scope('lstm_cells_layers'):
    crop_cell = tf.contrib.rnn.MultiRNNCell([lstm_cell() for _ in range(layer_num)], state_is_tuple=True)
# **步骤3：用全零来初始化state
init_state = crop_cell.zero_state(batch_size, dtype=tf.float32)
# **步骤4：调用dynamic_rnn()来让我们构建好的网络运行起来
# ** 当 time_major==False 时， outputs.shape = [batch_size, timestep_size, hidden_size]
# ** 所以，可以取 h_state = outputs[:, -1, :] 作为最后输出
# ** state.shape = [layer_num, 2, batch_size, hidden_size],
# ** 或者，可以取 h_state = state[-1][1] 作为最后输出
# ** 最后输出维度是 [batch_size, hidden_size]
outputs, state = tf.nn.dynamic_rnn(crop_cell, inputs=model_inputs, initial_state=init_state, time_major=False)
h_state = state[-1][1]


####################################################################
# 上面LSTM输出是一个[hidden_size]的tensor，要分类的话，还需要接一个softmax层
# 首先定义 softmax 的连接权重矩阵和偏置
# out_W = tf.placeholder(tf.float32, [hidden_size, class_num], name='out_Weights')
# out_bias = tf.placeholder(tf.float32, [class_num], name='out_bias')
with tf.name_scope('weights'):
    out_weight = tf.Variable(tf.truncated_normal([hidden_size, class_num], stddev=0.1), dtype=tf.float32, name='weight')
    tf.summary.histogram('output_layer_weights', out_weight)
with tf.name_scope('biases'):
    out_bias = tf.Variable(tf.constant(0.2, shape=[class_num]), dtype=tf.float32, name='biase')
    tf.summary.histogram('output_layer_biases', out_bias)
# 输出预测
with tf.name_scope('output_layer'):
    y_predict = tf.nn.softmax(tf.matmul(h_state, out_weight) + out_bias)
    # y_predict = tf.nn.softsign(tf.matmul(h_state, out_weight) + out_bias)
    # y_predict = tf.nn.relu(tf.matmul(h_state, out_weight) + out_bias)
    tf.summary.histogram('outputs', y_predict)

####################################################################
# 损失和评估函数
with tf.name_scope('loss'):
    cross_entropy = -tf.reduce_mean(model_labels * tf.log(y_predict))
    # cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=y_predict, labels=model_labels))
    tf.summary.scalar('loss', cross_entropy)
# 优化
with tf.name_scope('train'):
    train_op = tf.train.AdamOptimizer(lr).minimize(cross_entropy)
# 精度
with tf.name_scope('accuracy'):
    correct_prediction = tf.equal(tf.argmax(y_predict, 1), tf.argmax(model_labels, 1))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))
    tf.summary.scalar('accuracy', accuracy)

merged = tf.summary.merge_all()

####################################################################
# 会话训练+预测
with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    train_writer = tf.summary.FileWriter(tbdir, sess.graph)
    for i in range(max_iterations):
        batch_x, batch_y = data_utils.sample_batch(train_x, train_y, batch_size_num)
        if (i+1) % 200 == 0:
            train_accuracy, summery_merged = sess.run([accuracy, merged], feed_dict={model_inputs: batch_x, model_labels: batch_y, keep_prob: 1.0, batch_size: batch_size_num})
            print("Step %d, training accuracy %g" % ((i+1), train_accuracy))
            train_writer.add_summary(summery_merged, i+1)

        sess.run(train_op, feed_dict={model_inputs: batch_x, model_labels: batch_y, keep_prob: 0.5, batch_size: batch_size_num})

    ####################################################################
    # 回话预测
    test_results = sess.run(y_predict, feed_dict={model_inputs: test_x, keep_prob: 1.0, batch_size: np.shape(test_x)[0]})
    np.savetxt('F:/experiment-dataset/03-lstm-sar-crop/data/test_results22.csv', test_results, delimiter=',')

    train_writer.close()

print("LSTM-Crop Over!!!")

####################################################################
####################################################################
####################################################################
