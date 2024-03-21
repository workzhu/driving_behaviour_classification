import numpy as np
import tensorflow as tf



def LSTM_Network(feature_mat, config):
    """定义一个LSTM网络，
       包含2层LSTM层，每层有n_hidden=32个cell，
       和1个输出层，是一个全连接层
       
      参数:
        feature_mat: ndarray 特征矩阵，形状为[batch_size, time_steps, n_inputs]
        config: 包含网络配置的类
        
      返回:
              : 矩阵 输出形状为 [batch_size, n_classes]
    """
    # 将输入矩阵转置为 [time_steps, batch_size, n_inputs]
    feature_mat = tf.transpose(feature_mat, [1, 0, 2])

    # 将输入矩阵重塑为 [batch_size * time_steps, n_inputs]
    feature_mat = tf.reshape(feature_mat, [-1, config.n_inputs], name="features_reshape")

    # 应用隐藏层变换: relu(W * features + biases)
    hidden = tf.nn.relu(tf.matmul(
        feature_mat, config.W['hidden']
    ) + config.biases['hidden'])

    # 将重塑后的矩阵拆分为一个张量列表，每个张量是一个时间步
    hidden = tf.split(0, config.n_steps, hidden, name="input_hidden")

    # 定义一个具有n_hidden个单元和2个LSTM层的MultiRNNCell
    lstm_cell = tf.nn.rnn_cell.BasicLSTMCell(config.n_hidden, forget_bias=1.0)

    # 在隐藏层上应用RNN操作
    lsmt_layers = tf.nn.rnn_cell.MultiRNNCell([lstm_cell] * 2)

    # 将最后一个时间步的输出作为最终输出
    outputs, _ = tf.nn.rnn(lsmt_layers, hidden, dtype=tf.float32)
    lstm_last_output = outputs[-1]

    # 应用输出层变换: W * lstm_last_output + biases
    # Linear activation
    final_out = tf.add(tf.matmul(lstm_last_output, config.W['output']), config.biases['output'], name="logits")
    
    return final_out
